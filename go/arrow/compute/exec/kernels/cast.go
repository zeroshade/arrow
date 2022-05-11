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
	castdoc                                      = functions.FunctionDoc{}
	resolveOutputFromOpts functions.TypeResolver = func(ctx *functions.KernelCtx, args []compute.ValueDescr) (compute.ValueDescr, error) {
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
		ScalarFunction: functions.NewScalarFunction(name, functions.Unary(), nil),
		outID:          outType,
		inputIDs:       make([]arrow.Type, 0),
	}
}

func (c *CastFunction) AddKernel(inType arrow.Type, kernel functions.ScalarKernel) error {
	kernel.Init = func(kc *functions.KernelCtx, kia functions.KernelInitArgs) (functions.KernelState, error) {
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

func (c *CastFunction) AddNewKernel(inID arrow.Type, inTypes []functions.InputType, out functions.OutputType, kernelExec functions.ArrayKernelExec, nullHandling functions.NullHandling, memalloc functions.MemAlloc) error {
	kernel := functions.NewScalarKernel(inTypes, out, kernelExec, nil)
	kernel.MemAlloc = memalloc
	kernel.NullHandling = nullHandling
	return c.AddKernel(inID, kernel)
}

func (c *CastFunction) DispatchExact(vals []compute.ValueDescr) (functions.Kernel, error) {
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
		if arg0.Kind() == functions.ExactType {
			return k, nil
		}
	}

	return candidates[0], nil
}

func addCommonCasts(out arrow.Type, outType functions.OutputType, fn *CastFunction) {
	// from null
	kernel := functions.ScalarKernel{}
	kernel.Exec = castFromNull
	kernel.Signature = functions.NewKernelSig([]functions.InputType{functions.NewExactInput(arrow.Null, compute.ShapeAny)}, outType, false)
	kernel.NullHandling = functions.NullComputeNoPrealloc
	kernel.MemAlloc = functions.MemNoPrealloc
	err := fn.AddKernel(arrow.NULL, kernel)
	debug.Assert(err == nil, "failure adding cast from null kernel")

	if canCastFromDictionary(out) {
		err = fn.AddNewKernel(arrow.DICTIONARY, []functions.InputType{functions.NewInputIDType(arrow.DICTIONARY)}, outType, trivialScalarUnaryAsArrayExec(unpackDictionary, functions.NullIntersection),
			functions.NullComputeNoPrealloc, functions.MemNoPrealloc)
		debug.Assert(err == nil, "failed adding dictionary cast kernel")
	}

	err = fn.AddNewKernel(arrow.EXTENSION, []functions.InputType{functions.NewInputIDType(arrow.EXTENSION)}, outType, castFromExtension, functions.NullComputeNoPrealloc, functions.MemNoPrealloc)
	debug.Assert(err == nil, "failed adding extension cast kernel")
}

func generateVarBinaryBaseStandin(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	return nil
}

func castFunctorStandin(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	return nil
}

func castFromNull(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	if batch.Values[0].Kind() != compute.KindScalar {
		output := out.(*compute.ArrayDatum).Value
		arr := scalar.MakeArrayOfNull(output.DataType(), int(batch.Length), ctx.Ctx.Allocator())
		output.Release()
		arr.Data().Retain()
		out.(*compute.ArrayDatum).Value = arr.Data()
	}
	return nil
}

func castFromExtension(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	// opts := ctx.State().(*compute.CastOptions)
	if batch.Values[0].Kind() == compute.KindScalar {

	}

	return errors.New("not implemented")
}

func addZeroCopyCast(in arrow.Type, intype functions.InputType, outType functions.OutputType, fn *CastFunction) {
	sig := functions.NewKernelSig([]functions.InputType{intype}, outType, false)
	kernel := functions.NewScalarKernelWithSig(sig, trivialScalarUnaryAsArrayExec(internal.ZeroCopyCastExec, functions.NullIntersection), nil)
	kernel.NullHandling = functions.NullComputeNoPrealloc
	kernel.MemAlloc = functions.MemNoPrealloc
	fn.AddKernel(in, kernel)
}

func addSimpleCast(inTyp functions.InputType, outTyp functions.OutputType, fn *CastFunction) {

}

func RegisterScalarCasts(reg *functions.FunctionRegistry) {
	fn := functions.NewMetaFunction("cast", functions.Unary(), castdoc, func(ctx context.Context, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
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
