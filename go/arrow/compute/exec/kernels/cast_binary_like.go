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
	"fmt"
	"math"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/internal"
	"github.com/apache/arrow/go/v9/arrow/memory"
)

func fsbToFsb(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)
	input := batch.Values[0].(*compute.ArrayDatum).Value
	inputWidth := input.DataType().(*arrow.FixedSizeBinaryType).ByteWidth
	outputWidth := options.ToType.(*arrow.FixedSizeBinaryType).ByteWidth

	if inputWidth != outputWidth {
		return fmt.Errorf("%w: failed casting from %s to %s: widths must match", compute.ErrInvalid, input.DataType(), out.Type())
	}
	return internal.ZeroCopyCastExec(ctx, batch, out)
}

func binaryToBinary(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)
	input := batch.Values[0].(*compute.ArrayDatum).Value

	if input.DataType().ID() != arrow.STRING && out.Type().ID() == arrow.STRING && !options.AllowInvalidUtf8 {
		if err := internal.ValidateUtf8Binary(input); err != nil {
			return err
		}
	}
	// we only have 32-bit offsets implemented in go, so we don't need to
	// worry yet about casting the offsets
	return internal.ZeroCopyCastExec(ctx, batch, out)
}

func fsbToBinary(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
	options := ctx.State.(*compute.CastOptions)
	input := batch.Values[0].(*compute.ArrayDatum).Value
	outarr := out.(*compute.ArrayDatum).Value

	if outarr.DataType().ID() == arrow.STRING && !options.AllowInvalidUtf8 {
		if err := internal.ValidateUtf8FixedSizeBinary(input); err != nil {
			return err
		}
	}

	// check for offset overflow
	width := input.DataType().(*arrow.FixedSizeBinaryType).ByteWidth
	maxOffset := width * input.Len()
	if maxOffset > math.MaxInt32 {
		return fmt.Errorf("%w: failed casting from %s to %s: input array too large", compute.ErrInvalid, input.DataType(), out.Type())
	}

	// sweet. copy buffers then generate indices
	offset := outarr.Offset()
	buffers := []*memory.Buffer{nil, outarr.Buffers()[1], input.Buffers()[1]}
	if input.Offset() == outarr.Offset() {
		buffers[0] = input.Buffers()[0]
	} else {
		offset = 0
		buffers[0] = ctx.AllocateBitmap(int64(input.Len()))
		bitutil.CopyBitmap(input.Buffers()[0].Bytes(), input.Offset(), input.Len(), buffers[0].Bytes(), 0)
	}
	outarr.Reset(outarr.DataType(), input.Len(), buffers, nil, input.NullN(), offset)

	offsetBytes := outarr.Buffers()[1].Bytes()
	offsets := arrow.Int32Traits.CastFromBytes(offsetBytes)
	offsets[0] = int32(input.Offset() * width)
	for i := 0; i < input.Len(); i++ {
		offsets[i+1] = offsets[i] + int32(width)
	}
	return nil
}

func addNumberStringCasts(fn *CastFunction, outType functions.OutputType) {
	fn.AddNewKernel(arrow.BOOL, []functions.InputType{functions.NewExactInput(arrow.FixedWidthTypes.Boolean, compute.ShapeAny)}, outType,
		trivialScalarUnaryAsArrayExec(internal.GenerateNumericToString(arrow.BOOL), functions.NullComputeNoPrealloc),
		functions.NullIntersection, functions.MemNoPrealloc)

	for _, intyp := range numericTypes {
		fn.AddNewKernel(intyp.ID(), []functions.InputType{functions.NewExactInput(intyp, compute.ShapeAny)}, outType,
			trivialScalarUnaryAsArrayExec(internal.GenerateNumericToString(intyp.ID()), functions.NullComputeNoPrealloc),
			functions.NullIntersection, functions.MemNoPrealloc)
	}
}

func addTemporalToStringCasts(fn *CastFunction, outType functions.OutputType) {
	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outType,
		trivialScalarUnaryAsArrayExec(internal.CastTimestampToString, functions.NullComputeNoPrealloc),
		functions.NullIntersection, functions.MemNoPrealloc)
	for _, inType := range []arrow.Type{arrow.DATE32, arrow.DATE64, arrow.TIME32, arrow.TIME64} {
		fn.AddNewKernel(inType, []functions.InputType{functions.NewInputIDType(inType)}, outType,
			trivialScalarUnaryAsArrayExec(internal.GenerateNumericToString(inType), functions.NullComputeNoPrealloc),
			functions.NullIntersection, functions.MemPrealloc)
	}
}

func addBinaryCast(fn *CastFunction, exec functions.ArrayKernelExec, outType functions.OutputType, inType arrow.Type) {
	fn.AddNewKernel(inType, []functions.InputType{functions.NewInputIDType(inType)}, outType,
		trivialScalarUnaryAsArrayExec(exec, functions.NullComputeNoPrealloc), functions.NullIntersection, functions.MemPrealloc)
}

func addBinaryToBinaryCast(fn *CastFunction, outType functions.OutputType) {
	addBinaryCast(fn, binaryToBinary, outType, arrow.STRING)
	addBinaryCast(fn, binaryToBinary, outType, arrow.BINARY)
	addBinaryCast(fn, fsbToBinary, outType, arrow.FIXED_SIZE_BINARY)
}

func getBinaryLikeCasts() (out []CastFunction) {
	castBinary := NewCastFunction("cast_binary", arrow.BINARY)
	binOut := functions.NewOutputType(arrow.BinaryTypes.Binary)
	addCommonCasts(arrow.BINARY, binOut, &castBinary)
	addBinaryToBinaryCast(&castBinary, binOut)

	castString := NewCastFunction("cast_string", arrow.STRING)
	strOut := functions.NewOutputType(arrow.BinaryTypes.String)
	addCommonCasts(arrow.STRING, strOut, &castString)
	addBinaryToBinaryCast(&castString, strOut)
	addNumberStringCasts(&castString, strOut)
	addTemporalToStringCasts(&castString, strOut)

	castFSB := NewCastFunction("cast_fixed_size_binary", arrow.FIXED_SIZE_BINARY)
	fsbOut := functions.NewOutputTypeResolver(resolveOutputFromOpts)
	addCommonCasts(arrow.FIXED_SIZE_BINARY, fsbOut, &castFSB)
	castFSB.AddNewKernel(arrow.FIXED_SIZE_BINARY, []functions.InputType{functions.NewInputIDType(arrow.FIXED_SIZE_BINARY)},
		functions.NewOutputTypeResolver(firstType),
		trivialScalarUnaryAsArrayExec(fsbToFsb, functions.NullComputeNoPrealloc), functions.NullIntersection, functions.MemPrealloc)

	return []CastFunction{castBinary, castString, castFSB}
}
