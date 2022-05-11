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
	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/internal"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"golang.org/x/exp/constraints"
)

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
		kn = internal.CastDecimal128ToInteger[uint8]
	case arrow.UINT16:
		kn = internal.CastDecimal128ToInteger[uint16]
	case arrow.UINT32:
		kn = internal.CastDecimal128ToInteger[uint32]
	case arrow.UINT64:
		kn = internal.CastDecimal128ToInteger[uint64]
	case arrow.INT8:
		kn = internal.CastDecimal128ToInteger[int8]
	case arrow.INT16:
		kn = internal.CastDecimal128ToInteger[int16]
	case arrow.INT32:
		kn = internal.CastDecimal128ToInteger[int32]
	case arrow.INT64:
		kn = internal.CastDecimal128ToInteger[int64]
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

var outputTargetType = functions.NewOutputTypeResolver(resolveOutputFromOpts)

func getCastToDecimal128() CastFunction {
	outType := outputTargetType

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
