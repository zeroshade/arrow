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

package internal

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"unsafe"

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/array"
	"github.com/apache/arrow/go/v8/arrow/compute"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v8/arrow/decimal128"
	"github.com/apache/arrow/go/v8/arrow/internal/debug"
	"github.com/apache/arrow/go/v8/arrow/scalar"
	"golang.org/x/exp/constraints"
)

type primitive interface {
	constraints.Integer | constraints.Float
}

func reinterpret[T primitive | ~bool | decimal128.Num](b []byte) (res []T) {
	var t T
	sz := int(unsafe.Sizeof(t))
	return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), len(b)/sz)
}

func getVals[T primitive | ~bool | decimal128.Num](arr arrow.ArrayData, buf int) []T {
	res := reinterpret[T](arr.Buffers()[buf].Bytes())
	return res[arr.Offset() : arr.Offset()+arr.Len()]
}

func DoStaticCast[In, Out primitive](in []In, out []Out) {
	for i := range in {
		out[i] = Out(in[i])
	}
}

func CastPrimitive[In, Out primitive](input, output compute.Datum) {
	if input.Kind() == compute.KindArray {
		arrIn := input.(*compute.ArrayDatum).Value
		arrOut := output.(*compute.ArrayDatum).Value
		in := getVals[In](arrIn, 1)
		out := getVals[Out](arrOut, 1)
		DoStaticCast(in, out)
		return
	}

	in := input.(*compute.ScalarDatum).Value.(scalar.PrimitiveScalar)
	out := output.(*compute.ScalarDatum).Value.(scalar.PrimitiveScalar)
	DoStaticCast(reinterpret[In](in.Data()), reinterpret[Out](out.Data()))
}

func CastPrimitiveMemCpy[T primitive](input, output compute.Datum) {
	if input.Kind() == compute.KindArray {
		arrIn := input.(*compute.ArrayDatum).Value
		arrOut := output.(*compute.ArrayDatum).Value
		in := getVals[T](arrIn, 1)
		out := getVals[T](arrOut, 1)
		copy(out, in)
		return
	}

	in := input.(*compute.ScalarDatum).Value.(scalar.PrimitiveScalar)
	out := output.(*compute.ScalarDatum).Value.(scalar.PrimitiveScalar)
	copy(out.Data(), in.Data())
}

func CastNumberMemCpy(typ arrow.Type, input, output compute.Datum) {
	switch typ {
	case arrow.INT8:
		CastPrimitiveMemCpy[int8](input, output)
	case arrow.INT16:
		CastPrimitiveMemCpy[int16](input, output)
	case arrow.INT32:
		CastPrimitiveMemCpy[int32](input, output)
	case arrow.INT64:
		CastPrimitiveMemCpy[int64](input, output)
	case arrow.UINT8:
		CastPrimitiveMemCpy[uint8](input, output)
	case arrow.UINT16:
		CastPrimitiveMemCpy[uint16](input, output)
	case arrow.UINT32:
		CastPrimitiveMemCpy[uint32](input, output)
	case arrow.UINT64:
		CastPrimitiveMemCpy[uint64](input, output)
	case arrow.FLOAT32:
		CastPrimitiveMemCpy[float32](input, output)
	case arrow.FLOAT64:
		CastPrimitiveMemCpy[float64](input, output)
	}
}

func CastNumberImpl[T primitive](outtype arrow.Type, input, output compute.Datum) {
	switch outtype {
	case arrow.INT8:
		CastPrimitive[T, int8](input, output)
	case arrow.INT16:
		CastPrimitive[T, int16](input, output)
	case arrow.INT32:
		CastPrimitive[T, int32](input, output)
	case arrow.INT64:
		CastPrimitive[T, int64](input, output)
	case arrow.UINT8:
		CastPrimitive[T, uint8](input, output)
	case arrow.UINT16:
		CastPrimitive[T, uint16](input, output)
	case arrow.UINT32:
		CastPrimitive[T, uint32](input, output)
	case arrow.UINT64:
		CastPrimitive[T, uint64](input, output)
	case arrow.FLOAT32:
		CastPrimitive[T, float32](input, output)
	case arrow.FLOAT64:
		CastPrimitive[T, float64](input, output)
	}
}

func CastNumberToNumberUnsafe(intype, outtype arrow.Type, input, output compute.Datum) {
	if intype == outtype {
		CastNumberMemCpy(intype, input, output)
		return
	}

	switch intype {
	case arrow.INT8:
		CastNumberImpl[int8](outtype, input, output)
	case arrow.INT16:
		CastNumberImpl[int16](outtype, input, output)
	case arrow.INT32:
		CastNumberImpl[int32](outtype, input, output)
	case arrow.INT64:
		CastNumberImpl[int64](outtype, input, output)
	case arrow.UINT8:
		CastNumberImpl[uint8](outtype, input, output)
	case arrow.UINT16:
		CastNumberImpl[uint16](outtype, input, output)
	case arrow.UINT32:
		CastNumberImpl[uint32](outtype, input, output)
	case arrow.UINT64:
		CastNumberImpl[uint64](outtype, input, output)
	case arrow.FLOAT32:
		CastNumberImpl[float32](outtype, input, output)
	case arrow.FLOAT64:
		CastNumberImpl[float64](outtype, input, output)
	}
}

func ZeroCopyCastExec(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	debug.Assert(batch.Values[0].Kind() == compute.KindArray, "invalid kind for zerocopycastexec")

	input := batch.Values[0].(*compute.ArrayDatum).Value
	output := out.(*compute.ArrayDatum).Value.(*array.Data)
	output.Reset(output.DataType(), input.Len(), input.Buffers(), input.Children(), input.NullN(), input.Offset())
	return nil
}

func toInteger[T constraints.Integer](allowOverflow bool, min, max, v decimal128.Num, err *error) T {
	if !allowOverflow && (v.Less(min) || max.Less(v)) {
		debug.Log("integer value out of bounds")
		*err = errors.New("integer value out of bounds")
		return T(0)
	}
	return T(v.LowBits())
}

func castDecimal128ToInteger[T constraints.Integer](ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum, min, max decimal128.Num) error {
	opts := ctx.State.(*compute.CastOptions)
	inputType := batch.Values[0].Type().(*arrow.Decimal128Type)
	inScale := inputType.Scale

	var exec functions.ArrayKernelExec
	if opts.AllowDecimalTruncate {
		if inScale < 0 {
			exec = scalarUnaryNotNullStateful(func(ctx *functions.KernelCtx, val decimal128.Num, err *error) T {
				v := val.IncreaseScaleBy(-inScale)
				return toInteger[T](opts.AllowIntOverflow, min, max, v, err)
			})
		} else {
			exec = scalarUnaryNotNullStateful(func(ctx *functions.KernelCtx, val decimal128.Num, err *error) T {
				v := val.ReduceScaleBy(inScale, true)
				return toInteger[T](opts.AllowIntOverflow, min, max, v, err)
			})
		}
	} else {
		exec = scalarUnaryNotNullStateful(func(ctx *functions.KernelCtx, val decimal128.Num, err *error) T {
			v, e := val.Rescale(inScale, 0)
			if e != nil {
				*err = e
				return T(0)
			}
			return toInteger[T](opts.AllowIntOverflow, min, max, v, err)
		})
	}
	return exec(ctx, batch, out)
}

func CastDecimal128ToIntegerUnsigned[T constraints.Unsigned](ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	var (
		nbits = unsafe.Sizeof(T(0)) << 3
		min   = decimal128.FromU64(0)
		max   = decimal128.FromU64(uint64((1 << nbits) - 1))
	)
	return castDecimal128ToInteger[T](ctx, batch, out, min, max)
}

func CastDecimal128ToIntegerSigned[T constraints.Signed](ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	var (
		nbits  = unsafe.Sizeof(T(0)) << 3
		minVal = -(1 << (nbits - 1))
		min    = decimal128.FromI64(int64(minVal))
		max    = decimal128.FromI64(int64(^minVal))
	)
	return castDecimal128ToInteger[T](ctx, batch, out, min, max)
}

func CastIntegerToDecimal128[T constraints.Integer](ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	outType := out.Type().(*arrow.Decimal128Type)
	outScale := outType.Scale
	outPrecision := outType.Precision

	if outScale < 0 {
		return errors.New("scale must be non-negative")
	}
	precision, err := maxDecimalDigitsForInt(batch.Values[0].Type().ID())
	if err != nil {
		return err
	}
	precision += outScale
	if outPrecision < precision {
		return fmt.Errorf("precision is not great enough for result. It should be at least %d", precision)
	}

	inType := batch.Values[0].Type().ID()
	var getDecimal func(v T) decimal128.Num
	switch inType {
	case arrow.UINT8, arrow.UINT16, arrow.UINT32, arrow.UINT64:
		getDecimal = func(v T) decimal128.Num { return decimal128.FromU64(uint64(v)) }
	default:
		getDecimal = func(v T) decimal128.Num { return decimal128.FromI64(int64(v)) }
	}
	exec := scalarUnaryNotNullStateful(func(ctx *functions.KernelCtx, val T, err *error) decimal128.Num {
		out, er := getDecimal(val).Rescale(0, outScale)
		if er != nil {
			*err = er
		}
		return out
	})
	return exec(ctx, batch, out)
}

func CastDecimal128ToDecimal128(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)

	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	outType := out.Type().(*arrow.Decimal128Type)

	inScale := inType.Scale
	outScale := outType.Scale

	var exec functions.ArrayKernelExec
	if options.AllowDecimalTruncate {
		if inScale < outScale {
			exec = scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, val decimal128.Num, _ *error) decimal128.Num {
				return val.IncreaseScaleBy(outScale - inScale)
			})
		} else {
			exec = scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, val decimal128.Num, _ *error) decimal128.Num {
				return val.ReduceScaleBy(inScale-outScale, true)
			})
		}
		return exec(ctx, batch, out)
	}
	exec = scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, val decimal128.Num, err *error) decimal128.Num {
		out, e := val.Rescale(inScale, outScale)
		if e != nil {
			*err = e
			return decimal128.Num{}
		}

		if out.FitsInPrecision(outType.Precision) {
			return out
		}

		*err = errors.New("invalid precision for values")
		return decimal128.Num{}
	})
	return exec(ctx, batch, out)
}

func CastDecimalToFloat(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	inScale := inType.Scale

	exec := scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, v decimal128.Num, err *error) float32 {
		out := (&big.Float{}).SetInt(v.BigInt())
		out.Quo(out, big.NewFloat(math.Pow10(int(inScale))))
		res, _ := out.Float32()
		return res
	})
	return exec(ctx, batch, out)
}

func CastDecimalToDouble(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	inScale := inType.Scale

	exec := scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, v decimal128.Num, _ *error) float64 {
		out := (&big.Float{}).SetInt(v.BigInt())
		out.Quo(out, big.NewFloat(math.Pow10(int(inScale))))
		res, _ := out.Float64()
		return res
	})
	return exec(ctx, batch, out)
}

func FloatToDecimal128(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	// options := ctx.State.(*compute.CastOptions)
	outType := out.Type().(*arrow.Decimal128Type)
	outScale := outType.Scale
	outPrec := outType.Precision

	exec := scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, v float32, err *error) decimal128.Num {
		out, e := decimal128.FromFloat32(v, outPrec, outScale)
		if e != nil {
			*err = e
		}
		return out
	})
	return exec(ctx, batch, out)
}

func DoubleToDecimal128(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	// options := ctx.State.(*compute.CastOptions)
	outType := out.Type().(*arrow.Decimal128Type)
	outScale := outType.Scale
	outPrec := outType.Precision

	exec := scalarUnaryNotNullStateful(func(_ *functions.KernelCtx, v float64, err *error) decimal128.Num {
		out, e := decimal128.FromFloat64(v, outPrec, outScale)
		if e != nil {
			*err = e
		}
		return out

	})
	return exec(ctx, batch, out)
}
