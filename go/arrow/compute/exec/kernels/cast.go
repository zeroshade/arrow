// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kernels

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/internal"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/arrow/scalar"
)

var (
	castTable             map[arrow.Type]*CastFunction
	castInit              sync.Once
	castdoc                                    = compute.FunctionDoc{}
	resolveOutputFromOpts compute.TypeResolver = func(ctx *compute.KernelCtx, args []compute.ValueDescr) (compute.ValueDescr, error) {
		options := ctx.State.(*compute.CastOptions)
		return compute.ValueDescr{Type: options.ToType, Shape: args[0].Shape}, nil
	}
)

func addCastFuncs(funcs []CastFunction) {
	for i, f := range funcs {
		castTable[f.outID] = &funcs[i]
	}
}

func initCastTable() {
	castInit.Do(func() {
		castTable = make(map[arrow.Type]*CastFunction)
		addCastFuncs(getNumericCasts())
		addCastFuncs(getTemporalCasts())
		addCastFuncs(getBinaryLikeCasts())
	})
}

func GetCastFunction(toType arrow.DataType) (*CastFunction, error) {
	initCastTable()
	fn, ok := castTable[toType.ID()]
	if !ok {
		return nil, fmt.Errorf("unsupported cast to %s (no available cast function for type)", toType)
	}
	return fn, nil
}

func CanCast(from, to arrow.DataType) bool {
	initCastTable()
	fn, ok := castTable[to.ID()]
	if !ok {
		return false
	}

	debug.Assert(fn.outID == to.ID(), "type ids should match")
	for _, id := range fn.inputIDs {
		if from.ID() == id {
			return true
		}
	}
	return false
}

type CastFunction struct {
	functions.ScalarFunction

	inputIDs []arrow.Type
	outID    arrow.Type
}

func NewCastFunction(name string, outType arrow.Type) CastFunction {
	return CastFunction{
		ScalarFunction: functions.NewScalarFunction(name, compute.Unary(), nil),
		outID:          outType,
		inputIDs:       make([]arrow.Type, 0),
	}
}

func (c *CastFunction) AddKernel(inType arrow.Type, kernel functions.ScalarKernel) error {
	kernel.Init = func(kc *compute.KernelCtx, kia compute.KernelInitArgs) (compute.KernelState, error) {
		if _, ok := kia.Options.(*compute.CastOptions); ok {
			return kia.Options, nil
		}
		return nil, errors.New("attempted to initialize KernelState from null options")
	}
	if err := c.ScalarFunction.AddKernel(kernel); err != nil {
		return err
	}
	c.inputIDs = append(c.inputIDs, inType)
	return nil
}

func (c *CastFunction) AddNewKernel(inID arrow.Type, inTypes []compute.InputType, out compute.OutputType, kernelExec functions.ArrayKernelExec, nullHandling compute.NullHandling, memalloc compute.MemAlloc) error {
	kernel := functions.NewScalarKernel(inTypes, out, kernelExec, nil)
	kernel.MemAlloc = memalloc
	kernel.NullHandling = nullHandling
	return c.AddKernel(inID, kernel)
}

func (c *CastFunction) DispatchBest(vals []compute.ValueDescr) (compute.Kernel, error) {
	return c.DispatchExact(vals)
}

func (c *CastFunction) DispatchExact(vals []compute.ValueDescr) (compute.Kernel, error) {
	if err := c.CheckArityDescr(vals); err != nil {
		return nil, err
	}

	candidates := make([]*functions.ScalarKernel, 0)
	kernels := c.Kernels()
	for i, k := range kernels {
		if k.Signature.MatchesInputs(vals) {
			candidates = append(candidates, &kernels[i])
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("unsupported cast from %s to %s using function %s",
			vals[0], c.outID, c.Name())
	}

	if len(candidates) == 1 {
		return candidates[0], nil
	}

	// we are in a casting scenario where we may have both an EXACT_TYPE and a
	// SAME_TYPE_ID. So we'll see if there is an exact match among the candidates
	// and if not we just return the first one
	for _, k := range candidates {
		arg0 := k.Signature.InputTypes()[0]
		if arg0.Kind() == compute.ExactType {
			return k, nil
		}
	}

	return candidates[0], nil
}

func addCommonCasts(out arrow.Type, outType compute.OutputType, fn *CastFunction) {
	// from null
	kernel := functions.ScalarKernel{}
	kernel.Exec = castFromNull
	kernel.Signature = compute.NewKernelSig([]compute.InputType{compute.NewExactInput(arrow.Null, compute.ShapeAny)}, outType, false)
	kernel.NullHandling = compute.NullComputeNoPrealloc
	kernel.MemAlloc = compute.MemNoPrealloc
	err := fn.AddKernel(arrow.NULL, kernel)
	debug.Assert(err == nil, "failure adding cast from null kernel")

	if canCastFromDictionary(out) {
		err = fn.AddNewKernel(arrow.DICTIONARY, []compute.InputType{compute.NewInputIDType(arrow.DICTIONARY)}, outType, trivialScalarUnaryAsArrayExec(unpackDictionary, compute.NullIntersection),
			compute.NullComputeNoPrealloc, compute.MemNoPrealloc)
		debug.Assert(err == nil, "failed adding dictionary cast kernel")
	}

	err = fn.AddNewKernel(arrow.EXTENSION, []compute.InputType{compute.NewInputIDType(arrow.EXTENSION)}, outType, castFromExtension, compute.NullComputeNoPrealloc, compute.MemNoPrealloc)
	debug.Assert(err == nil, "failed adding extension cast kernel")
}

func generateVarBinaryBaseStandin(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	return nil
}

func castFunctorStandin(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	return nil
}

func castFromNull(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	if batch.Values[0].Kind() != compute.KindScalar {
		output := out.(*compute.ArrayDatum).Value
		arr := scalar.MakeArrayOfNull(output.DataType(), int(batch.Length), ctx.Ctx.Mem)
		output.Release()
		arr.Data().Retain()
		out.(*compute.ArrayDatum).Value = arr.Data()
	}
	return nil
}

func castFromExtension(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	// opts := ctx.State().(*compute.CastOptions)
	if batch.Values[0].Kind() == compute.KindScalar {

	}

	return errors.New("not implemented")
}

func addZeroCopyCast(in arrow.Type, intype compute.InputType, outType compute.OutputType, fn *CastFunction) {
	sig := compute.NewKernelSig([]compute.InputType{intype}, outType, false)
	kernel := functions.NewScalarKernelWithSig(sig, trivialScalarUnaryAsArrayExec(internal.ZeroCopyCastExec, compute.NullIntersection), nil)
	kernel.NullHandling = compute.NullComputeNoPrealloc
	kernel.MemAlloc = compute.MemNoPrealloc
	fn.AddKernel(in, kernel)
}

func RegisterScalarCasts(reg *functions.FunctionRegistry) {
	fn := functions.NewMetaFunction("cast", compute.Unary(), castdoc, func(ctx context.Context, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
		castOpts, ok := opts.(*compute.CastOptions)
		if !ok || castOpts.ToType == nil {
			return nil, errors.New("cast requires options with a ToType")
		}

		if arrow.TypeEqual(args[0].Type(), castOpts.ToType) {
			return compute.NewDatum(args[0]), nil
		}

		fn, err := GetCastFunction(castOpts.ToType)
		if err != nil {
			return nil, err
		}
		return functions.ExecuteFunction(ctx, fn, args, castOpts)
	})
	reg.AddFunction(&fn, true)
}
