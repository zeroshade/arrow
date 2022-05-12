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
	"strconv"
	"time"
	"unsafe"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/array"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/decimal128"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/arrow/scalar"
	"github.com/apache/arrow/go/v9/internal/bitutils"
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

func getVals[T primitive | decimal128.Num](arr arrow.ArrayData, buf int) []T {
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

func ZeroCopyCastExec(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
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

func CastDecimal128ToInteger[T constraints.Integer](ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	var (
		opts       = ctx.State.(*compute.CastOptions)
		inputType  = batch.Values[0].Type().(*arrow.Decimal128Type)
		inScale    = inputType.Scale
		exec       functions.ArrayKernelExec
		minLowbits = uint64(MinOf[T]())
		minHiBits  int64
		max        = decimal128.FromU64(uint64(MaxOf[T]()))
	)
	if MinOf[T]() < 0 {
		minHiBits = -1
	}
	min := decimal128.New(minHiBits, minLowbits)
	if opts.AllowDecimalTruncate {
		if inScale < 0 {
			exec = scalarUnaryNotNullStateful(func(ctx *compute.KernelCtx, val decimal128.Num, err *error) T {
				v := val.IncreaseScaleBy(-inScale)
				return toInteger[T](opts.AllowIntOverflow, min, max, v, err)
			})
		} else {
			exec = scalarUnaryNotNullStateful(func(ctx *compute.KernelCtx, val decimal128.Num, err *error) T {
				v := val.ReduceScaleBy(inScale, true)
				return toInteger[T](opts.AllowIntOverflow, min, max, v, err)
			})
		}
	} else {
		exec = scalarUnaryNotNullStateful(func(ctx *compute.KernelCtx, val decimal128.Num, err *error) T {
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

func CastIntegerToDecimal128[T constraints.Integer](ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
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
	exec := scalarUnaryNotNullStateful(func(ctx *compute.KernelCtx, val T, err *error) decimal128.Num {
		out, er := getDecimal(val).Rescale(0, outScale)
		if er != nil {
			*err = er
		}
		return out
	})
	return exec(ctx, batch, out)
}

func CastDecimal128ToDecimal128(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)

	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	outType := out.Type().(*arrow.Decimal128Type)

	inScale := inType.Scale
	outScale := outType.Scale

	var exec functions.ArrayKernelExec
	if options.AllowDecimalTruncate {
		if inScale < outScale {
			exec = scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, val decimal128.Num, _ *error) decimal128.Num {
				return val.IncreaseScaleBy(outScale - inScale)
			})
		} else {
			exec = scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, val decimal128.Num, _ *error) decimal128.Num {
				return val.ReduceScaleBy(inScale-outScale, true)
			})
		}
		return exec(ctx, batch, out)
	}
	exec = scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, val decimal128.Num, err *error) decimal128.Num {
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

func CastDecimalToFloat(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	inScale := inType.Scale

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, v decimal128.Num, err *error) float32 {
		return v.ToFloat32(inScale)
	})
	return exec(ctx, batch, out)
}

func CastDecimalToDouble(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.Decimal128Type)
	inScale := inType.Scale

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, v decimal128.Num, _ *error) float64 {
		return v.ToFloat64(inScale)
	})
	return exec(ctx, batch, out)
}

func FloatToDecimal128(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)
	outType := out.Type().(*arrow.Decimal128Type)
	outScale := outType.Scale
	outPrec := outType.Precision

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, v float32, err *error) decimal128.Num {
		out, e := decimal128.FromFloat32(v, outPrec, outScale)
		if !options.AllowDecimalTruncate && e != nil {
			*err = e
			return decimal128.Num{}
		}
		return out
	})
	return exec(ctx, batch, out)
}

func DoubleToDecimal128(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)
	outType := out.Type().(*arrow.Decimal128Type)
	outScale := outType.Scale
	outPrec := outType.Precision

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, v float64, err *error) decimal128.Num {
		out, e := decimal128.FromFloat64(v, outPrec, outScale)
		if !options.AllowDecimalTruncate && e != nil {
			*err = e
			return decimal128.Num{}
		}
		return out

	})
	return exec(ctx, batch, out)
}

func GetParseStringExec(out arrow.Type) functions.ArrayKernelExec {
	switch out {
	case arrow.INT8:
		return parseStringToNumberImpl(func(s string) (int8, error) {
			v, err := strconv.ParseInt(s, 0, 8)
			return int8(v), err
		})
	case arrow.INT16:
		return parseStringToNumberImpl(func(s string) (int16, error) {
			v, err := strconv.ParseInt(s, 0, 16)
			return int16(v), err
		})
	case arrow.INT32:
		return parseStringToNumberImpl(func(s string) (int32, error) {
			v, err := strconv.ParseInt(s, 0, 32)
			return int32(v), err
		})
	case arrow.INT64:
		return parseStringToNumberImpl(func(s string) (int64, error) {
			return strconv.ParseInt(s, 0, 64)
		})
	case arrow.UINT8:
		return parseStringToNumberImpl(func(s string) (uint8, error) {
			v, err := strconv.ParseUint(s, 0, 8)
			return uint8(v), err
		})
	case arrow.UINT16:
		return parseStringToNumberImpl(func(s string) (uint16, error) {
			v, err := strconv.ParseUint(s, 0, 16)
			return uint16(v), err
		})
	case arrow.UINT32:
		return parseStringToNumberImpl(func(s string) (uint32, error) {
			v, err := strconv.ParseUint(s, 0, 32)
			return uint32(v), err
		})
	case arrow.UINT64:
		return parseStringToNumberImpl(func(s string) (uint64, error) {
			return strconv.ParseUint(s, 0, 64)
		})
	case arrow.FLOAT32:
		return parseStringToNumberImpl(func(s string) (float32, error) {
			v, err := strconv.ParseFloat(s, 32)
			return float32(v), err
		})
	case arrow.FLOAT64:
		return parseStringToNumberImpl(func(s string) (float64, error) {
			return strconv.ParseFloat(s, 64)
		})
	}
	return nil
}

func parseStringToNumberImpl[T constraints.Integer | constraints.Float](parseFn func(string) (T, error)) functions.ArrayKernelExec {
	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		exec := scalarUnaryNotNullStatefulBinaryArg(func(_ *compute.KernelCtx, input []byte, err *error) T {
			st := *(*string)(unsafe.Pointer(&input))
			v, e := parseFn(st)
			if e != nil {
				*err = e
			}
			return v
		})
		return exec(ctx, batch, out)
	}
}

var (
	epoch = time.Unix(0, 0)
)

func StringToTimestamp(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	outType := out.Type().(*arrow.TimestampType)
	exec := scalarUnaryNotNullStatefulBinaryArg(func(_ *compute.KernelCtx, input []byte, err *error) arrow.Timestamp {
		v := *(*string)(unsafe.Pointer(&input))
		result, e := arrow.TimestampFromString(v, outType.Unit)
		if e != nil {
			*err = e
		}
		return result
	})
	return exec(ctx, batch, out)
}

func TimestampToDate32(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.TimestampType)
	debug.Assert(out.Type().ID() == arrow.DATE32, "timestamptoDate32 called with type other than Date32 as output")

	fnToTime, err := inType.GetToTimeFunc()
	if err != nil {
		return err
	}

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, arg0 arrow.Timestamp, err *error) arrow.Date32 {
		tm := fnToTime(arg0)
		return arrow.Date32FromTime(tm)
	})
	return exec(ctx, batch, out)
}

func TimestampToDate64(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.TimestampType)
	debug.Assert(out.Type().ID() == arrow.DATE64, "timestamptoDate64 called with type other than Date64 as output")

	fnToTime, err := inType.GetToTimeFunc()
	if err != nil {
		return err
	}

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, arg0 arrow.Timestamp, err *error) arrow.Date64 {
		tm := fnToTime(arg0)
		return arrow.Date64FromTime(tm)
	})
	return exec(ctx, batch, out)
}

func TimestampToTime32(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.TimestampType)
	outType := out.Type().(*arrow.Time32Type)

	options := ctx.State.(*compute.CastOptions)

	fnToTime, err := inType.GetToTimeFunc()
	if err != nil {
		return err
	}
	if inType.TimeZone != "" && inType.TimeZone != "UTC" {
		origFn := fnToTime
		fnToTime = func(t arrow.Timestamp) time.Time {
			v := origFn(t)
			_, offset := v.Zone()
			return v.Add(time.Duration(offset) * time.Second).UTC()
		}
	}

	var fn func(time.Duration, *error) arrow.Time32

	switch outType.Unit {
	case arrow.Second:
		fn = func(d time.Duration, _ *error) arrow.Time32 {
			return arrow.Time32(d.Seconds())
		}
	case arrow.Millisecond:
		fn = func(d time.Duration, _ *error) arrow.Time32 {
			return arrow.Time32(d.Milliseconds())
		}
	default:
		return fmt.Errorf("%w: bad unit type for cast to time32: %s", compute.ErrInvalid, outType.Unit)
	}

	op, factor := arrow.GetTimestampConvert(inType.Unit, outType.Unit)
	if op == arrow.ConvDIVIDE && !options.AllowTimeTruncate {
		origFn := fn
		switch inType.Unit {
		case arrow.Millisecond:
			fn = func(d time.Duration, err *error) arrow.Time32 {
				v := origFn(d, err)
				if int64(v)*factor != d.Milliseconds() {
					*err = fmt.Errorf("%w: cast would lose data: %d", compute.ErrInvalid, d.Milliseconds())
				}
				return v
			}
		case arrow.Microsecond:
			fn = func(d time.Duration, err *error) arrow.Time32 {
				v := origFn(d, err)
				if int64(v)*factor != d.Microseconds() {
					*err = fmt.Errorf("%w: cast would lose data: %d", compute.ErrInvalid, d.Microseconds())
				}
				return v
			}
		case arrow.Nanosecond:
			fn = func(d time.Duration, err *error) arrow.Time32 {
				v := origFn(d, err)
				if int64(v)*factor != d.Nanoseconds() {
					*err = fmt.Errorf("%w: cast would lose data: %d", compute.ErrInvalid, d.Nanoseconds())
				}
				return v
			}
		}
	}

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, arg0 arrow.Timestamp, err *error) arrow.Time32 {
		t := fnToTime(arg0)
		dur := t.Sub(t.Truncate(24 * time.Hour))
		return fn(dur, err)
	})
	return exec(ctx, batch, out)
}

func TimestampToTime64(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	inType := batch.Values[0].Type().(*arrow.TimestampType)
	outType := out.Type().(*arrow.Time64Type)

	options := ctx.State.(*compute.CastOptions)

	fnToTime, err := inType.GetToTimeFunc()
	if err != nil {
		return err
	}

	if inType.TimeZone != "" && inType.TimeZone != "UTC" {
		origFn := fnToTime
		fnToTime = func(t arrow.Timestamp) time.Time {
			v := origFn(t)
			_, offset := v.Zone()
			return v.Add(time.Duration(offset) * time.Second).UTC()
		}
	}

	var fn func(time.Duration, *error) arrow.Time64

	op, _ := arrow.GetTimestampConvert(inType.Unit, outType.Unit)
	if op == arrow.ConvDIVIDE && !options.AllowTimeTruncate {
		// only one case can happen here, microseconds. since nanoseconds
		// wouldn't be a downscale
		fn = func(d time.Duration, err *error) arrow.Time64 {
			if d.Nanoseconds() != d.Microseconds()*int64(time.Microsecond) {
				*err = fmt.Errorf("%w: cast would lose data: %d", compute.ErrInvalid, d.Nanoseconds())
			}
			return arrow.Time64(d.Microseconds())
		}
	} else {
		switch outType.Unit {
		case arrow.Microsecond:
			fn = func(d time.Duration, _ *error) arrow.Time64 {
				return arrow.Time64(d.Microseconds())
			}
		case arrow.Nanosecond:
			fn = func(d time.Duration, _ *error) arrow.Time64 {
				return arrow.Time64(d.Nanoseconds())
			}
		default:
			return fmt.Errorf("%w: bad unit type for cast to time64: %s", compute.ErrInvalid, outType.Unit)
		}
	}

	exec := scalarUnaryNotNullStateful(func(_ *compute.KernelCtx, arg0 arrow.Timestamp, err *error) arrow.Time64 {
		t := fnToTime(arg0)
		dur := t.Sub(t.Truncate(24 * time.Hour))
		return fn(dur, err)
	})
	return exec(ctx, batch, out)
}

func SimpleTemporalCast[I, O arrow.Duration | arrow.Time32 | arrow.Time64 | arrow.Timestamp](ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	debug.Assert(batch.Values[0].Kind() == compute.KindArray, "duration to duration cast expects an array")

	input := batch.Values[0].(*compute.ArrayDatum).Value
	output := out.(*compute.ArrayDatum).Value

	// if units are the same, zero copy, otherwise convert
	inType := input.DataType().(arrow.TemporalWithUnit)
	outType := output.DataType().(arrow.TemporalWithUnit)
	if inType.TimeUnit() == outType.TimeUnit() && inType.BitWidth() == outType.BitWidth() {
		output.Reset(output.DataType(), input.Len(), input.Buffers(), input.Children(), input.NullN(), input.Offset())
		return nil
	}

	op, factor := arrow.GetTimestampConvert(inType.TimeUnit(), outType.TimeUnit())
	return ShiftTime[I, O](ctx, op, factor, input, output)
}

func date32ToString(bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[arrow.Date32](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(vals[pos].FormattedString())
		}, func() {
			bldr.AppendNull()
		})
}

func date64ToString(bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[arrow.Date64](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(vals[pos].FormattedString())
		}, func() {
			bldr.AppendNull()
		})
}

func time32ToString(bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[arrow.Time32](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	unit := input.DataType().(*arrow.Time32Type).Unit
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(vals[pos].FormattedString(unit))
		}, func() {
			bldr.AppendNull()
		})
}

func time64ToString(bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[arrow.Time64](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	unit := input.DataType().(*arrow.Time64Type).Unit
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(vals[pos].FormattedString(unit))
		}, func() {
			bldr.AppendNull()
		})
}

func numericToStringSigned[T constraints.Signed](bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[T](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(strconv.FormatInt(int64(vals[pos]), 10))
		}, func() {
			bldr.AppendNull()
		})
}

func numericToStringUnsigned[T constraints.Unsigned](bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[T](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(strconv.FormatUint(uint64(vals[pos]), 10))
		}, func() {
			bldr.AppendNull()
		})
}

func numericToStringFloat[T constraints.Float](bldr *array.BinaryBuilder, input arrow.ArrayData) {
	vals := getVals[T](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitsize := 32
	if input.DataType().ID() == arrow.FLOAT64 {
		bitsize = 64
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(strconv.FormatFloat(float64(vals[pos]), 'g', -1, bitsize))
		}, func() {
			bldr.AppendNull()
		})
}

func numericToStringBool(bldr *array.BinaryBuilder, input arrow.ArrayData) {
	offset := input.Offset()
	data := input.Buffers()[1].Bytes()
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(strconv.FormatBool(bitutil.BitIsSet(data, offset+int(pos))))
		}, func() { bldr.AppendNull() })
}

func GenerateNumericToString(typ arrow.Type) functions.ArrayKernelExec {
	var runfn func(*array.BinaryBuilder, arrow.ArrayData)

	switch typ {
	case arrow.BOOL:
		runfn = numericToStringBool
	case arrow.INT8:
		runfn = numericToStringSigned[int8]
	case arrow.INT16:
		runfn = numericToStringSigned[int16]
	case arrow.INT32:
		runfn = numericToStringSigned[int32]
	case arrow.INT64:
		runfn = numericToStringSigned[int64]
	case arrow.UINT8:
		runfn = numericToStringUnsigned[uint8]
	case arrow.UINT16:
		runfn = numericToStringUnsigned[uint16]
	case arrow.UINT32:
		runfn = numericToStringUnsigned[uint32]
	case arrow.UINT64:
		runfn = numericToStringUnsigned[uint64]
	case arrow.FLOAT32:
		runfn = numericToStringFloat[float32]
	case arrow.FLOAT64:
		runfn = numericToStringFloat[float64]
	case arrow.DATE32:
		runfn = date32ToString
	case arrow.DATE64:
		runfn = date64ToString
	case arrow.TIME32:
		runfn = time32ToString
	case arrow.TIME64:
		runfn = time64ToString
	}

	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		input := batch.Values[0].(*compute.ArrayDatum).Value
		output := out.(*compute.ArrayDatum).Value

		bldr := array.NewBinaryBuilder(ctx.Ctx.Mem, output.DataType().(arrow.BinaryDataType))
		defer bldr.Release()

		runfn(bldr, input)

		result := bldr.NewArray()
		defer result.Release()
		output.Reset(result.DataType(), result.Len(), result.Data().Buffers(), nil, result.NullN(), result.Data().Offset())
		return nil
	}
}

func CastTimestampToString(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	input := batch.Values[0].(*compute.ArrayDatum).Value
	output := out.(*compute.ArrayDatum).Value

	bldr := array.NewBinaryBuilder(ctx.Ctx.Mem, output.DataType().(arrow.BinaryDataType))
	defer bldr.Release()

	inputType := input.DataType().(*arrow.TimestampType)
	toTime, err := inputType.GetToTimeFunc()
	if err != nil {
		return err
	}

	// prealloc
	fmtstring := "2006-01-02 15:04:05"
	switch inputType.Unit {
	case arrow.Millisecond:
		fmtstring += ".000"
	case arrow.Microsecond:
		fmtstring += ".000000"
	case arrow.Nanosecond:
		fmtstring += ".000000000"
	}

	switch inputType.TimeZone {
	case "UTC":
		fmtstring += "Z"
	case "":
	default:
		fmtstring += "-0700"
	}

	strlen := len(fmtstring)
	bldr.Reserve(input.Len())
	bldr.ReserveData((input.Len() - input.NullN()) * strlen)

	vals := getVals[arrow.Timestamp](input, 1)
	var bitmap []byte
	if input.Buffers()[0] != nil {
		bitmap = input.Buffers()[0].Bytes()
	}
	bitutils.VisitBitBlocks(bitmap, int64(input.Offset()), int64(input.Len()),
		func(pos int64) {
			bldr.AppendString(toTime(vals[pos]).Format(fmtstring))
		}, func() { bldr.AppendNull() })

	result := bldr.NewArray()
	defer result.Release()
	output.Reset(result.DataType(), result.Len(), result.Data().Buffers(), nil, result.NullN(), result.Data().Offset())
	return nil
}
