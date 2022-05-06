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

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/bitutil"
	"github.com/apache/arrow/go/v8/arrow/compute"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v8/arrow/decimal128"
	"github.com/apache/arrow/go/v8/arrow/scalar"
	"github.com/apache/arrow/go/v8/internal/bitutils"
)

func unboxScalar[T primitive | ~bool | decimal128.Num](v scalar.Scalar) T {
	return reinterpret[T](v.(scalar.PrimitiveScalar).Data())[0]
}

func boxScalar[T primitive | ~bool | decimal128.Num](val T, v scalar.Scalar) {
	reinterpret[T](v.(scalar.PrimitiveScalar).Data())[0] = val
}

func ExecScalarUnary[OutType, Arg0Type primitive](op func(ctx *functions.KernelCtx, val Arg0Type) OutType) functions.ArrayKernelExec {
	execArray := func(ctx *functions.KernelCtx, arg0 arrow.ArrayData, out compute.Datum) error {
		arg := getVals[Arg0Type](arg0, 1)
		outData := out.(*compute.ArrayDatum).Value
		output := getVals[OutType](outData, 1)

		for i := range arg {
			output[i] = op(ctx, arg[i])
		}
		return nil
	}

	execScalar := func(ctx *functions.KernelCtx, arg0 scalar.Scalar, out compute.Datum) error {
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

	return func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
		switch batch.Values[0].Kind() {
		case compute.KindArray:
			return execArray(ctx, batch.Values[0].(*compute.ArrayDatum).Value, out)
		default:
			return execScalar(ctx, batch.Values[0].(*compute.ScalarDatum).Value, out)
		}
	}
}

func ExecScalarUnaryBoolArg[OutType primitive](op func(ctx *functions.KernelCtx, val bool) OutType) functions.ArrayKernelExec {
	execArray := func(ctx *functions.KernelCtx, arg0 arrow.ArrayData, out compute.Datum) error {
		rdr := bitutil.NewBitmapReader(arg0.Buffers()[1].Bytes(), arg0.Offset(), arg0.Len())
		outData := out.(*compute.ArrayDatum).Value
		output := getVals[OutType](outData, 1)

		for i := 0; i < rdr.Len(); i++ {
			output[i] = op(ctx, rdr.Set())
			rdr.Next()
		}

		return nil
	}

	execScalar := func(ctx *functions.KernelCtx, arg0 scalar.Scalar, out compute.Datum) error {
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

	return func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
		switch batch.Values[0].Kind() {
		case compute.KindArray:
			return execArray(ctx, batch.Values[0].(*compute.ArrayDatum).Value, out)
		default:
			return execScalar(ctx, batch.Values[0].(*compute.ScalarDatum).Value, out)
		}
	}
}

func scalarUnaryNotNullStateful[OutType primitive | decimal128.Num, Arg0Type primitive | decimal128.Num](op func(*functions.KernelCtx, Arg0Type, *error) OutType) functions.ArrayKernelExec {
	return func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
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
			bitutils.VisitBitBlocks(arg0.Buffers()[0].Bytes(), int64(arg0.Offset()), int64(arg0.Len()),
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
