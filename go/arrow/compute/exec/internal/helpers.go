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
	"fmt"
	"math"
	"unicode/utf8"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/decimal128"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/arrow/scalar"
	"github.com/apache/arrow/go/v9/internal/bitutils"
	"golang.org/x/exp/constraints"
)

func unboxScalar[T primitive | ~bool | decimal128.Num](v scalar.Scalar) T {
	return reinterpret[T](v.(scalar.PrimitiveScalar).Data())[0]
}

func boxScalar[T primitive | ~bool | decimal128.Num](val T, v scalar.Scalar) {
	reinterpret[T](v.(scalar.PrimitiveScalar).Data())[0] = val
}

func SizeOf[T constraints.Integer]() uint {
	x := uint16(1 << 8)
	y := uint32(2 << 16)
	z := uint64(4 << 32)
	return 1 + uint(T(x))>>8 + uint(T(y))>>16 + uint(T(z))>>32
}

func MinOf[T constraints.Integer]() T {
	if ones := ^T(0); ones < 0 {
		return ones << (8*SizeOf[T]() - 1)
	}
	return 0
}

func MaxOf[T constraints.Integer]() T {
	ones := ^T(0)
	if ones < 0 {
		return ones ^ (ones << (8*SizeOf[T]() - 1))
	}
	return ones
}

func ExecScalarUnary[OutType, Arg0Type primitive](op func(ctx *compute.KernelCtx, val Arg0Type) OutType) functions.ArrayKernelExec {
	execArray := func(ctx *compute.KernelCtx, arg0 arrow.ArrayData, out compute.Datum) error {
		arg := getVals[Arg0Type](arg0, 1)
		outData := out.(*compute.ArrayDatum).Value
		output := getVals[OutType](outData, 1)

		for i := range arg {
			output[i] = op(ctx, arg[i])
		}
		return nil
	}

	execScalar := func(ctx *compute.KernelCtx, arg0 scalar.Scalar, out compute.Datum) error {
		outScalar := out.(*compute.ScalarDatum).Value
		if arg0.IsValid() {
			outScalar.SetValid(true)
			val := unboxScalar[Arg0Type](arg0)
			outVal := op(ctx, val)
			// if err != nil {
			// 	return err
			// }
			boxScalar(outVal, outScalar)
			return nil
		}

		outScalar.SetValid(false)
		return nil
	}

	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		switch batch.Values[0].Kind() {
		case compute.KindArray:
			return execArray(ctx, batch.Values[0].(*compute.ArrayDatum).Value, out)
		default:
			return execScalar(ctx, batch.Values[0].(*compute.ScalarDatum).Value, out)
		}
	}
}

func ExecScalarUnaryBoolArg[OutType primitive](op func(ctx *compute.KernelCtx, val bool) OutType) functions.ArrayKernelExec {
	execArray := func(ctx *compute.KernelCtx, arg0 arrow.ArrayData, out compute.Datum) error {
		rdr := bitutil.NewBitmapReader(arg0.Buffers()[1].Bytes(), arg0.Offset(), arg0.Len())
		outData := out.(*compute.ArrayDatum).Value
		output := getVals[OutType](outData, 1)

		for i := 0; i < rdr.Len(); i++ {
			output[i] = op(ctx, rdr.Set())
			rdr.Next()
		}

		return nil
	}

	execScalar := func(ctx *compute.KernelCtx, arg0 scalar.Scalar, out compute.Datum) error {
		outScalar := out.(*compute.ScalarDatum).Value
		if arg0.IsValid() {
			outScalar.SetValid(true)
			val := arg0.(*scalar.Boolean).Value
			outVal := op(ctx, val)
			// if err != nil {
			// 	return err
			// }
			boxScalar(outVal, outScalar)
			return nil
		}

		outScalar.SetValid(false)
		return nil
	}

	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		switch batch.Values[0].Kind() {
		case compute.KindArray:
			return execArray(ctx, batch.Values[0].(*compute.ArrayDatum).Value, out)
		default:
			return execScalar(ctx, batch.Values[0].(*compute.ScalarDatum).Value, out)
		}
	}
}

func scalarUnaryNotNullStateful[OutType primitive | decimal128.Num, Arg0Type primitive | decimal128.Num](op func(*compute.KernelCtx, Arg0Type, *error) OutType) functions.ArrayKernelExec {
	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		if batch.Values[0].Kind() == compute.KindArray {
			var (
				outArr   = out.(*compute.ArrayDatum).Value
				outData  = getVals[OutType](outArr, 1)
				outPos   = 0
				arg0     = batch.Values[0].(*compute.ArrayDatum).Value
				arg0Data = getVals[Arg0Type](arg0, 1)
				def      OutType
				err      error
			)
			var bitmap []byte
			if arg0.Buffers()[0] != nil {
				bitmap = arg0.Buffers()[0].Bytes()
			}

			bitutils.VisitBitBlocks(bitmap, int64(arg0.Offset()), int64(arg0.Len()),
				func(pos int64) {
					outData[outPos] = op(ctx, arg0Data[pos], &err)
					outPos++
				}, func() {
					outData[outPos] = def
					outPos++
				})
			return err
		}
		var (
			arg0      = batch.Values[0].(*compute.ScalarDatum).Value
			outScalar = out.(*compute.ScalarDatum).Value
			err       error
		)
		if arg0.IsValid() {
			arg0Val := unboxScalar[Arg0Type](arg0)
			boxScalar(op(ctx, arg0Val, &err), outScalar)
		}
		return err
	}
}

var emptyData = []byte{0}

func ValidateUtf8FixedSizeBinary(data arrow.ArrayData) error {
	debug.Assert(data.DataType().ID() == arrow.FIXED_SIZE_BINARY, "validateUtf8FixedSizeBinary expects a fixed sized binary type")
	width := int64(data.DataType().(*arrow.FixedSizeBinaryType).ByteWidth)
	var bitmap []byte
	if data.Buffers()[0] != nil {
		bitmap = data.Buffers()[0].Bytes()
	}

	rawData := data.Buffers()[1].Bytes()

	return bitutils.VisitBitBlocksShort(bitmap, int64(data.Offset()), int64(data.Len()),
		func(pos int64) error {
			pos += int64(data.Offset())
			beg := pos * width
			end := (pos + 1) * width
			if !utf8.Valid(rawData[beg:end]) {
				return fmt.Errorf("%w: invalid utf8 bytes %x", compute.ErrInvalid, rawData[beg:end])
			}
			return nil
		}, func() error { return nil })
}

func ValidateUtf8Binary(data arrow.ArrayData) error {
	debug.Assert(data.DataType().ID() == arrow.BINARY || data.DataType().ID() == arrow.STRING, "validateutf8binary expects a variable length binary type")
	offsets := reinterpret[int32](data.Buffers()[1].Bytes())[data.Offset() : data.Offset()+data.Len()+1]
	var inputData []byte
	if data.Buffers()[2] != nil {
		inputData = data.Buffers()[2].Bytes()
	} else {
		inputData = emptyData
	}
	var bitmap []byte
	if data.Buffers()[0] != nil {
		bitmap = data.Buffers()[0].Bytes()
	}
	return bitutils.VisitBitBlocksShort(bitmap, int64(data.Offset()), int64(data.Len()),
		func(pos int64) error {
			v := inputData[offsets[pos]:offsets[pos+1]]
			if !utf8.Valid(v) {
				return fmt.Errorf("%w: invalid UTF8 bytes: %x", compute.ErrInvalid, v)
			}
			return nil
		}, func() error {
			return nil
		})
}

func scalarUnaryNotNullStatefulBinaryArg[OutType primitive | decimal128.Num](op func(*compute.KernelCtx, []byte, *error) OutType) functions.ArrayKernelExec {
	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		if batch.Values[0].Kind() == compute.KindArray {
			var (
				outArr      = out.(*compute.ArrayDatum).Value
				outData     = getVals[OutType](outArr, 1)
				outPos      = 0
				arg0        = batch.Values[0].(*compute.ArrayDatum).Value
				arg0Offsets = reinterpret[int32](arg0.Buffers()[1].Bytes())[arg0.Offset() : arg0.Offset()+arg0.Len()+1]
				def         OutType
				err         error
				arg0Data    []byte
			)
			if arg0.Buffers()[2] != nil {
				arg0Data = arg0.Buffers()[2].Bytes()
			} else {
				arg0Data = emptyData
			}
			var bitmap []byte
			if arg0.Buffers()[0] != nil {
				bitmap = arg0.Buffers()[0].Bytes()
			}

			bitutils.VisitBitBlocks(bitmap, int64(arg0.Offset()), int64(arg0.Len()),
				func(pos int64) {
					v := arg0Data[arg0Offsets[pos]:arg0Offsets[pos+1]]
					outData[outPos] = op(ctx, v, &err)
					outPos++
				}, func() {
					outData[outPos] = def
					outPos++
				})
			return err
		}
		var (
			arg0      = batch.Values[0].(*compute.ScalarDatum).Value
			outScalar = out.(*compute.ScalarDatum).Value
			err       error
		)
		if arg0.IsValid() {
			boxScalar(op(ctx, arg0.(scalar.BinaryScalar).Data(), &err), outScalar)
		}
		return err
	}
}

func maxDecimalDigitsForInt(id arrow.Type) (int32, error) {
	switch id {
	case arrow.INT8, arrow.UINT8:
		return 3, nil
	case arrow.INT16, arrow.UINT16:
		return 5, nil
	case arrow.INT32, arrow.UINT32:
		return 10, nil
	case arrow.INT64:
		return 19, nil
	case arrow.UINT64:
		return 20, nil
	}
	return -1, fmt.Errorf("not an integer type: %s", id)
}

func ShiftTime[InT, OutT constraints.Integer](ctx *compute.KernelCtx, op arrow.TimestampConvertOP, factor int64, input, output arrow.ArrayData) error {
	options := ctx.State.(*compute.CastOptions)
	inData := getVals[InT](input, 1)
	outData := getVals[OutT](output, 1)

	switch {
	case factor == 1:
		for i, v := range inData {
			outData[i] = OutT(v)
		}
	case op == arrow.ConvMULTIPLY:
		if options.AllowTimeOverflow {
			for i, v := range inData {
				outData[i] = OutT(v) * OutT(factor)
			}
			break
		}

		maxVal, minVal := math.MaxInt64/factor, math.MinInt64/factor
		if input.NullN() == 0 {
			for i, v := range inData {
				if int64(v) < minVal || int64(v) > maxVal {
					return fmt.Errorf("%w: casting from %s to %s would result in out of bounds timestamp: %d",
						compute.ErrInvalid, input.DataType(), output.DataType(), v)
				}
				outData[i] = OutT(v) * OutT(factor)
			}
			break
		}

		bitrdr := bitutil.NewBitmapReader(input.Buffers()[0].Bytes(), input.Offset(), input.Len())
		for i, v := range inData {
			if bitrdr.Set() && (int64(v) < minVal || int64(v) > maxVal) {
				return fmt.Errorf("%w: casting from %s to %s would result in out of bounds timestamp: %d",
					compute.ErrInvalid, input.DataType(), output.DataType(), v)
			}
			outData[i] = OutT(v) * OutT(factor)
			bitrdr.Next()
		}
	default: // divide
		if options.AllowTimeTruncate {
			for i, v := range inData {
				outData[i] = OutT(v / InT(factor))
			}
			break
		}

		if input.NullN() == 0 {
			for i, v := range inData {
				outData[i] = OutT(v / InT(factor))
				if outData[i]*OutT(factor) != OutT(v) {
					return fmt.Errorf("%w: casting from %s to %s would lose data: %d",
						compute.ErrInvalid, input.DataType(), output.DataType(), v)
				}
			}
			break
		}

		bitrdr := bitutil.NewBitmapReader(input.Buffers()[0].Bytes(), input.Offset(), input.Len())
		for i, v := range inData {
			outData[i] = OutT(v / InT(factor))
			if bitrdr.Set() && (outData[i]*OutT(factor) != OutT(v)) {
				return fmt.Errorf("%w: casting from %s to %s would lose data: %d",
					compute.ErrInvalid, input.DataType(), output.DataType(), v)
			}
			bitrdr.Next()
		}
	}
	return nil
}
