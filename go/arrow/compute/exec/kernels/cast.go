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

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/compute"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/internal"
	"github.com/apache/arrow/go/v8/arrow/internal/debug"
	"github.com/apache/arrow/go/v8/arrow/scalar"
	"golang.org/x/exp/constraints"
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

func addCommonNumberCasts[T constraints.Integer | constraints.Float](out arrow.DataType, fn *CastFunction) {
	outtype := functions.NewOutputType(out)
	addCommonCasts(out.ID(), outtype, fn)

	err := fn.AddNewKernel(arrow.BOOL, []functions.InputType{functions.NewExactInput(arrow.FixedWidthTypes.Boolean, compute.ShapeAny)}, outtype, internal.ExecScalarUnaryBoolArg(func(_ *functions.KernelCtx, val bool) T {
		if val {
			return 1
		}
		return 0
	}), functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed adding bool number cast kernel")

	for _, in := range baseBinaryTypes {
		err = fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outtype, nil, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed adding basebinary cast kernel")
	}
}

func generateVarBinaryBaseStandin(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	return nil
}

func castFunctorStandin(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	return nil
}

func castFloatingToFloating(_ *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	internal.CastNumberToNumberUnsafe(batch.Values[0].Type().ID(), out.Type().ID(), batch.Values[0], out)
	return nil
}

func castIntegerToInteger(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	opts := ctx.State.(*compute.CastOptions)
	if !opts.AllowIntOverflow {
		if err := internal.IntsCanFit(batch.Values[0], out.Type()); err != nil {
			return err
		}
	}
	internal.CastNumberToNumberUnsafe(batch.Values[0].Type().ID(), out.Type().ID(), batch.Values[0], out)
	return nil
}

func castIntegerToFloating(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	opts := ctx.State.(*compute.CastOptions)
	outType := out.Type().ID()
	if !opts.AllowFloatTruncate {
		if err := internal.CheckIntToFloatTrunc(batch.Values[0], outType); err != nil {
			return err
		}
	}
	internal.CastNumberToNumberUnsafe(batch.Values[0].Type().ID(), outType, batch.Values[0], out)
	return nil
}

func castFloatingToInteger(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	opts := ctx.State.(*compute.CastOptions)
	internal.CastNumberToNumberUnsafe(batch.Values[0].Type().ID(), out.Type().ID(), batch.Values[0], out)
	if !opts.AllowFloatTruncate {
		if err := internal.CheckFloatToIntTrunc(batch.Values[0], out); err != nil {
			return err
		}
	}
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

func getCastToInt[T constraints.Integer](name string, outType arrow.DataType) CastFunction {
	fn := NewCastFunction(name, outType.ID())
	outputType := functions.NewOutputType(outType)
	for _, in := range intTypes {
		err := fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outputType, castIntegerToInteger, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed adding cast kernel")
	}

	for _, in := range floatingTypes {
		err := fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outputType, castFloatingToInteger, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed adding cast kernel")
	}

	addCommonNumberCasts[T](outType, &fn)
	addCastFromDecimal128(&fn, outType)
	return fn
}

func addCastFromDecimal128(fn *CastFunction, outType arrow.DataType) {
	outputType := functions.NewOutputType(outType)
	var kn functions.ArrayKernelExec
	switch outType.ID() {
	case arrow.UINT8:
		kn = internal.CastDecimal128ToIntegerUnsigned[uint8]
	case arrow.UINT16:
		kn = internal.CastDecimal128ToIntegerUnsigned[uint16]
	case arrow.UINT32:
		kn = internal.CastDecimal128ToIntegerUnsigned[uint32]
	case arrow.UINT64:
		kn = internal.CastDecimal128ToIntegerUnsigned[uint64]
	case arrow.INT8:
		kn = internal.CastDecimal128ToIntegerSigned[int8]
	case arrow.INT16:
		kn = internal.CastDecimal128ToIntegerSigned[int16]
	case arrow.INT32:
		kn = internal.CastDecimal128ToIntegerSigned[int32]
	case arrow.INT64:
		kn = internal.CastDecimal128ToIntegerSigned[int64]
	}
	err := fn.AddNewKernel(arrow.DECIMAL128, []functions.InputType{functions.NewInputIDType(arrow.DECIMAL128)}, outputType, kn, functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed adding decimal number cast kernel")
}

func getCastToFloating[T constraints.Float](name string, outType arrow.DataType) CastFunction {
	fn := NewCastFunction(name, outType.ID())
	outputType := functions.NewOutputType(outType)
	// casts from integer to floats
	for _, in := range intTypes {
		err := fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outputType, castIntegerToFloating, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed adding int to floating cast kernels")
	}

	for _, in := range floatingTypes {
		err := fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outputType, castFloatingToFloating, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed adding float to float cast kernels")
	}

	addCommonNumberCasts[T](outType, &fn)
	var exec functions.ArrayKernelExec
	switch outType.ID() {
	case arrow.FLOAT32:
		exec = internal.CastDecimalToFloat
	case arrow.FLOAT64:
		exec = internal.CastDecimalToDouble
	}

	err := fn.AddNewKernel(arrow.DECIMAL128, []functions.InputType{functions.NewInputIDType(arrow.DECIMAL128)}, outputType, exec, functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed adding decimal to float cast kernel")
	return fn
}

func addZeroCopyCast(in arrow.Type, intype functions.InputType, outType functions.OutputType, fn *CastFunction) {
	sig := functions.NewKernelSig([]functions.InputType{intype}, outType, false)
	kernel := functions.NewScalarKernelWithSig(sig, trivialScalarUnaryAsArrayExec(internal.ZeroCopyCastExec, functions.NullIntersection), nil)
	kernel.NullHandling = functions.NullComputeNoPrealloc
	kernel.MemAlloc = functions.MemNoPrealloc
	fn.AddKernel(in, kernel)
}

func getCastToDecimal128() CastFunction {
	outType := functions.NewOutputTypeResolver(resolveOutputFromOpts)

	fn := NewCastFunction("cast_decimal", arrow.DECIMAL128)
	addCommonCasts(arrow.DECIMAL128, outType, &fn)

	err := fn.AddNewKernel(arrow.FLOAT32, []functions.InputType{functions.NewExactInput(arrow.PrimitiveTypes.Float32, compute.ShapeAny)}, outType, internal.FloatToDecimal128, functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed adding cast float to decimal kernel")
	err = fn.AddNewKernel(arrow.FLOAT64, []functions.InputType{functions.NewExactInput(arrow.PrimitiveTypes.Float64, compute.ShapeAny)}, outType, internal.DoubleToDecimal128, functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed to add cast double to decimal kernel")

	for _, in := range intTypes {
		var exec functions.ArrayKernelExec
		switch in.ID() {
		case arrow.INT8:
			exec = internal.CastIntegerToDecimal128[int8]
		case arrow.UINT8:
			exec = internal.CastIntegerToDecimal128[uint8]
		case arrow.INT16:
			exec = internal.CastIntegerToDecimal128[int16]
		case arrow.UINT16:
			exec = internal.CastIntegerToDecimal128[uint16]
		case arrow.INT32:
			exec = internal.CastIntegerToDecimal128[int32]
		case arrow.UINT32:
			exec = internal.CastIntegerToDecimal128[uint32]
		case arrow.INT64:
			exec = internal.CastIntegerToDecimal128[int64]
		case arrow.UINT64:
			exec = internal.CastIntegerToDecimal128[uint64]
		}

		err = fn.AddNewKernel(in.ID(), []functions.InputType{functions.NewExactInput(in, compute.ShapeAny)}, outType, exec, functions.NullIntersection, functions.MemPrealloc)
		debug.Assert(err == nil, "failed to add cast integer to decimal kernel")
	}

	err = fn.AddNewKernel(arrow.DECIMAL128, []functions.InputType{functions.NewInputIDType(arrow.DECIMAL128)}, outType, internal.CastDecimal128ToDecimal128, functions.NullIntersection, functions.MemPrealloc)
	debug.Assert(err == nil, "failed to add cast decimal to decimal kernel")
	return fn
}

func getNumericCasts() (out []CastFunction) {
	out = make([]CastFunction, 0, 10)
	out = append(out, NewCastFunction("cast_null", arrow.NULL))
	out = append(out, getCastToInt[int8]("cast_int8", arrow.PrimitiveTypes.Int8))
	out = append(out, getCastToInt[int16]("cast_int16", arrow.PrimitiveTypes.Int16))
	castInt32 := getCastToInt[int32]("cast_int32", arrow.PrimitiveTypes.Int32)
	addZeroCopyCast(arrow.DATE32, functions.NewExactInput(arrow.FixedWidthTypes.Date32, compute.ShapeAny), functions.NewOutputType(arrow.PrimitiveTypes.Int32), &castInt32)
	addZeroCopyCast(arrow.TIME32, functions.NewInputIDType(arrow.TIME32), functions.NewOutputType(arrow.PrimitiveTypes.Int32), &castInt32)
	out = append(out, castInt32)

	castInt64 := getCastToInt[int64]("cast_int64", arrow.PrimitiveTypes.Int64)
	addZeroCopyCast(arrow.DATE64, functions.NewInputIDType(arrow.DATE64), functions.NewOutputType(arrow.PrimitiveTypes.Int64), &castInt64)
	addZeroCopyCast(arrow.DURATION, functions.NewInputIDType(arrow.DURATION), functions.NewOutputType(arrow.PrimitiveTypes.Int64), &castInt64)
	addZeroCopyCast(arrow.TIMESTAMP, functions.NewInputIDType(arrow.TIMESTAMP), functions.NewOutputType(arrow.PrimitiveTypes.Int64), &castInt64)
	addZeroCopyCast(arrow.TIME64, functions.NewInputIDType(arrow.TIME64), functions.NewOutputType(arrow.PrimitiveTypes.Int64), &castInt64)
	out = append(out, castInt64)

	out = append(out, getCastToInt[uint8]("cast_uint8", arrow.PrimitiveTypes.Uint8))
	out = append(out, getCastToInt[uint16]("cast_uint16", arrow.PrimitiveTypes.Uint16))
	out = append(out, getCastToInt[uint32]("cast_uint32", arrow.PrimitiveTypes.Uint32))
	out = append(out, getCastToInt[uint64]("cast_uint64", arrow.PrimitiveTypes.Uint64))

	// float16 is a special child
	castHalfFloat := NewCastFunction("cast_half_float", arrow.FLOAT16)
	addCommonCasts(arrow.FLOAT16, functions.NewOutputType(arrow.FixedWidthTypes.Float16), &castHalfFloat)
	out = append(out, castHalfFloat)

	out = append(out, getCastToFloating[float32]("cast_float", arrow.PrimitiveTypes.Float32))
	out = append(out, getCastToFloating[float64]("cast_double", arrow.PrimitiveTypes.Float64))

	out = append(out, getCastToDecimal128())
	return out
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
