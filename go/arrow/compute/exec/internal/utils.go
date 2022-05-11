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

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/internal/bitutils"
	"golang.org/x/exp/constraints"
)

func IntegersInRange[T constraints.Integer](datum compute.Datum, lowerBound, upperBound T) error {
	outOfBoundsMaybeNull := func(val T, valid bool) bool {
		return valid && (val < lowerBound || val > upperBound)
	}
	outOfBounds := func(val T) bool {
		return val < lowerBound || val > upperBound
	}
	getError := func(val T) error {
		return fmt.Errorf("integer value: %d not in range: [%d, %d]", val, lowerBound, upperBound)
	}

	if datum.Kind() == compute.KindScalar {
		scalarDatum := datum.(*compute.ScalarDatum)
		val := unboxScalar[T](scalarDatum.Value)
		if outOfBoundsMaybeNull(val, scalarDatum.Value.IsValid()) {
			return getError(val)
		}
		return nil
	}

	indices := datum.(*compute.ArrayDatum).Value
	data := getVals[T](indices, 1)
	var bitmap []byte
	if indices.Buffers()[0] != nil {
		bitmap = indices.Buffers()[0].Bytes()
	}

	bitCounter := bitutils.NewOptionalBitBlockCounter(bitmap, int64(indices.Offset()), int64(indices.Len()))
	pos := 0
	offsetPos := indices.Offset()
	for pos < indices.Len() {
		block := bitCounter.NextBlock()
		blockOutOfBounds := false
		if block.Popcnt == block.Len {
			// fast path! no branches!
			i := 0
			for chunk := int16(0); chunk < block.Len/8; chunk++ {
				for j := 0; j < 8; j++ {
					blockOutOfBounds = blockOutOfBounds || outOfBounds(data[i])
					i++
				}
			}
			for ; i < int(block.Len); i++ {
				blockOutOfBounds = blockOutOfBounds || outOfBounds(data[i])
			}
		} else if block.Popcnt > 0 {
			// there are nulls, only boundscheck non-null values
			i := 0
			for chunk := int16(0); chunk < block.Len/8; chunk++ {
				for j := 0; j < 8; j++ {
					blockOutOfBounds = blockOutOfBounds ||
						outOfBoundsMaybeNull(data[i], bitutil.BitIsSet(bitmap, offsetPos+i))
					i++
				}
			}
			for ; i < int(block.Len); i++ {
				blockOutOfBounds = blockOutOfBounds ||
					outOfBoundsMaybeNull(data[i], bitutil.BitIsSet(bitmap, offsetPos+i))
			}
		}
		if blockOutOfBounds {
			if indices.NullN() > 0 {
				for i := 0; i < int(block.Len); i++ {
					if outOfBoundsMaybeNull(data[i], bitutil.BitIsSet(bitmap, offsetPos+i)) {
						return getError(data[i])
					}
				}
			} else {
				for i := 0; i < int(block.Len); i++ {
					if outOfBounds(data[i]) {
						return getError(data[i])
					}
				}
			}
		}

		data = data[block.Len:]
		pos += int(block.Len)
		offsetPos += int(block.Len)
	}
	return nil
}

func getSafeMinSameSign[I, O constraints.Integer]() I {
	if SizeOf[I]() > SizeOf[O]() {
		return I(MinOf[O]())
	}
	return MinOf[I]()
}

func getSafeMaxSameSign[I, O constraints.Integer]() I {
	if SizeOf[I]() > SizeOf[O]() {
		return I(MaxOf[O]())
	}
	return MaxOf[I]()
}

func getSafeMaxSignedUnsigned[I constraints.Signed, O constraints.Unsigned]() I {
	if SizeOf[I]() <= SizeOf[O]() {
		return MaxOf[I]()
	}
	return I(MaxOf[O]())
}

func getSafeMaxUnsignedSigned[I constraints.Unsigned, O constraints.Signed]() I {
	if SizeOf[I]() < SizeOf[O]() {
		return MaxOf[I]()
	}
	return I(MaxOf[O]())
}

func getSafeMinMaxSigned[T constraints.Signed](target arrow.Type) (min, max T) {
	switch target {
	case arrow.UINT8:
		min = 0
		max = getSafeMaxSignedUnsigned[T, uint8]()
	case arrow.UINT16:
		min = 0
		max = getSafeMaxSignedUnsigned[T, uint16]()
	case arrow.UINT32:
		min = 0
		max = getSafeMaxSignedUnsigned[T, uint32]()
	case arrow.UINT64:
		min = 0
		max = getSafeMaxSignedUnsigned[T, uint64]()
	case arrow.INT8:
		min = getSafeMinSameSign[T, int8]()
		max = getSafeMaxSameSign[T, int8]()
	case arrow.INT16:
		min = getSafeMinSameSign[T, int16]()
		max = getSafeMaxSameSign[T, int16]()
	case arrow.INT32:
		min = getSafeMinSameSign[T, int32]()
		max = getSafeMaxSameSign[T, int32]()
	case arrow.INT64:
		min = getSafeMinSameSign[T, int64]()
		max = getSafeMaxSameSign[T, int64]()
	}
	return
}

func getSafeMinMaxUnsigned[T constraints.Unsigned](target arrow.Type) (min, max T) {
	min = 0
	switch target {
	case arrow.UINT8:
		max = getSafeMaxSameSign[T, uint8]()
	case arrow.UINT16:
		max = getSafeMaxSameSign[T, uint16]()
	case arrow.UINT32:
		max = getSafeMaxSameSign[T, uint32]()
	case arrow.UINT64:
		max = getSafeMaxSameSign[T, uint64]()
	case arrow.INT8:
		max = getSafeMaxUnsignedSigned[T, int8]()
	case arrow.INT16:
		max = getSafeMaxUnsignedSigned[T, int16]()
	case arrow.INT32:
		max = getSafeMaxUnsignedSigned[T, int32]()
	case arrow.INT64:
		max = getSafeMaxUnsignedSigned[T, int64]()
	}
	return
}

func IntsCanFit(datum compute.Datum, targetType arrow.DataType) error {
	if !arrow.IsInteger(targetType.ID()) {
		return fmt.Errorf("target type is not an integer type %s", targetType)
	}

	switch datum.Type().ID() {
	case arrow.INT8:
		min, max := getSafeMinMaxSigned[int8](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.UINT8:
		min, max := getSafeMinMaxUnsigned[uint8](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.INT16:
		min, max := getSafeMinMaxSigned[int16](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.UINT16:
		min, max := getSafeMinMaxUnsigned[uint16](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.INT32:
		min, max := getSafeMinMaxSigned[int32](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.UINT32:
		min, max := getSafeMinMaxUnsigned[uint32](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.INT64:
		min, max := getSafeMinMaxSigned[int64](targetType.ID())
		return IntegersInRange(datum, min, max)
	case arrow.UINT64:
		min, max := getSafeMinMaxUnsigned[uint64](targetType.ID())
		return IntegersInRange(datum, min, max)
	default:
		return errors.New("invalid type for boundschecking")
	}
}

func checkFloatTrunc[InT constraints.Float, OutT constraints.Integer](input, output compute.Datum) error {
	wasTrunc := func(out OutT, in InT) bool {
		return InT(out) != in
	}
	wasTruncMaybeNull := func(out OutT, in InT, valid bool) bool {
		return valid && InT(out) != in
	}
	getError := func(val InT) error {
		return fmt.Errorf("float value %f was truncated converting to %s", val, output.Type())
	}

	if input.Kind() == compute.KindScalar {
		debug.Assert(output.Kind() == compute.KindScalar, "attempting to convert from scalar to non-scalar output")
		inScalarVal := unboxScalar[InT](input.(*compute.ScalarDatum).Value)
		outScalar := output.(*compute.ScalarDatum).Value
		outScalarVal := unboxScalar[OutT](outScalar)
		if wasTruncMaybeNull(outScalarVal, inScalarVal, outScalar.IsValid()) {
			return getError(inScalarVal)
		}
		return nil
	}

	inarr := input.(*compute.ArrayDatum).Value
	outarr := output.(*compute.ArrayDatum).Value

	inData := getVals[InT](inarr, 1)
	outData := getVals[OutT](outarr, 1)
	var bitmap []byte
	if inarr.Buffers()[0] != nil {
		bitmap = inarr.Buffers()[0].Bytes()
	}

	bitCounter := bitutils.NewOptionalBitBlockCounter(bitmap, int64(inarr.Offset()), int64(inarr.Len()))
	pos := 0
	offsetPos := inarr.Offset()
	for pos < inarr.Len() {
		block := bitCounter.NextBlock()
		blockTrunc := false
		if block.Popcnt == block.Len {
			// fast path, no branches
			for i := 0; i < int(block.Len); i++ {
				blockTrunc = blockTrunc || wasTrunc(outData[i], inData[i])
			}
		} else if block.Popcnt > 0 {
			// there are nulls, boundscheck only non-null vals
			for i := 0; i < int(block.Len); i++ {
				blockTrunc = blockTrunc ||
					wasTruncMaybeNull(outData[i], inData[i], bitutil.BitIsSet(bitmap, offsetPos+i))
			}
		}
		if blockTrunc {
			if inarr.NullN() > 0 {
				for i := 0; i < int(block.Len); i++ {
					if wasTruncMaybeNull(outData[i], inData[i], bitutil.BitIsSet(bitmap, offsetPos+i)) {
						return getError(inData[i])
					}
				}
			} else {
				for i := 0; i < int(block.Len); i++ {
					if wasTrunc(outData[i], inData[i]) {
						return getError(inData[i])
					}
				}
			}
		}
		inData = inData[block.Len:]
		outData = outData[block.Len:]
		pos += int(block.Len)
		offsetPos += int(block.Len)
	}
	return nil
}

func checkFloatToIntTruncImpl[T constraints.Float](input, output compute.Datum) error {
	switch output.Type().ID() {
	case arrow.INT8:
		return checkFloatTrunc[T, int8](input, output)
	case arrow.INT16:
		return checkFloatTrunc[T, int16](input, output)
	case arrow.INT32:
		return checkFloatTrunc[T, int32](input, output)
	case arrow.INT64:
		return checkFloatTrunc[T, int64](input, output)
	case arrow.UINT8:
		return checkFloatTrunc[T, uint8](input, output)
	case arrow.UINT16:
		return checkFloatTrunc[T, uint16](input, output)
	case arrow.UINT32:
		return checkFloatTrunc[T, uint32](input, output)
	case arrow.UINT64:
		return checkFloatTrunc[T, uint64](input, output)
	}
	debug.Assert(false, "float to int truncation only for integer output")
	return nil
}

func CheckFloatToIntTrunc(input, output compute.Datum) error {
	switch input.Type().ID() {
	case arrow.FLOAT32:
		return checkFloatToIntTruncImpl[float32](input, output)
	case arrow.FLOAT64:
		return checkFloatToIntTruncImpl[float64](input, output)
	}
	debug.Assert(false, "float to int truncation only for 32 and 64 bit float input")
	return nil
}

func CheckIntToFloatTrunc(input compute.Datum, outType arrow.Type) error {
	switch input.Type().ID() {
	// small integeras are all exactly representable as whole numbers
	case arrow.INT8, arrow.INT16, arrow.UINT8, arrow.UINT16:
		return nil
	case arrow.INT32:
		if outType == arrow.FLOAT64 {
			return nil
		}
		const limit = int32(1 << 24)
		return IntegersInRange(input, -limit, limit)
	case arrow.UINT32:
		if outType == arrow.FLOAT64 {
			return nil
		}
		return IntegersInRange(input, 0, uint32(1<<24))
	case arrow.INT64:
		if outType == arrow.FLOAT32 {
			const limit = int64(1 << 24)
			return IntegersInRange(input, -limit, limit)
		}
		const limit = int64(1 << 53)
		return IntegersInRange(input, -limit, limit)
	case arrow.UINT64:
		if outType == arrow.FLOAT32 {
			return IntegersInRange(input, 0, uint64(1<<53))
		}
		return IntegersInRange(input, 0, uint64(1<<53))
	}
	debug.Assert(false, "intToFloatTrunc should only be called with int input")
	return nil
}
