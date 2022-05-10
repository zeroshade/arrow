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

package kernels_test

import (
	"context"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/array"
	"github.com/apache/arrow/go/v8/arrow/bitutil"
	"github.com/apache/arrow/go/v8/arrow/compute"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v8/arrow/compute/exec/kernels"
	"github.com/apache/arrow/go/v8/arrow/decimal128"
	"github.com/apache/arrow/go/v8/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

var (
	numericTypes = []arrow.DataType{
		arrow.PrimitiveTypes.Uint8, arrow.PrimitiveTypes.Int8,
		arrow.PrimitiveTypes.Uint16, arrow.PrimitiveTypes.Int16,
		arrow.PrimitiveTypes.Uint32, arrow.PrimitiveTypes.Int32,
		arrow.PrimitiveTypes.Uint64, arrow.PrimitiveTypes.Int64,
		arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64,
	}
	integerTypes = []arrow.DataType{
		arrow.PrimitiveTypes.Uint8, arrow.PrimitiveTypes.Int8,
		arrow.PrimitiveTypes.Uint16, arrow.PrimitiveTypes.Int16,
		arrow.PrimitiveTypes.Uint32, arrow.PrimitiveTypes.Int32,
		arrow.PrimitiveTypes.Uint64, arrow.PrimitiveTypes.Int64,
	}
	dictIndexTypes  = integerTypes
	baseBinaryTypes = []arrow.DataType{
		arrow.BinaryTypes.Binary, arrow.BinaryTypes.String,
	}
)

func checkScalar(t *testing.T, ctx context.Context, fn functions.Function, inputs []compute.Datum, expected compute.Datum, opts compute.FunctionOptions) {
	out, err := functions.ExecuteFunction(ctx, fn, inputs, opts)
	assert.NoError(t, err)
	defer out.Release()

	var (
		outarr, exarr arrow.Array
	)
	if ex, ok := expected.(*compute.ArrayDatum); ok {
		outarr = out.(*compute.ArrayDatum).MakeArray()
		defer outarr.Release()
		exarr = ex.MakeArray()
		defer exarr.Release()
	}

	assert.Truef(t, expected.Equals(out), "%s(%s) = %s != %s. got: %s \n, expected: %s", fn.Name(), inputs, out, expected, outarr, exarr)
}

func checkScalarUnary(t *testing.T, ctx context.Context, fn functions.Function, input, expected compute.Datum, opts compute.FunctionOptions) {
	checkScalar(t, ctx, fn, []compute.Datum{input}, expected, opts)
}

func maskArrayWithNullsAt(mem memory.Allocator, input arrow.Array, indices []int) arrow.Array {
	masked := array.NewData(input.DataType(), input.Len(), append([]*memory.Buffer{}, input.Data().Buffers()...), input.Data().Children(), array.UnknownNullCount, input.Data().Offset())
	defer masked.Release()

	if masked.Buffers()[0] != nil {
		masked.Buffers()[0].Release()
	}
	masked.Buffers()[0] = memory.NewResizableBuffer(mem)
	masked.Buffers()[0].Resize(int(bitutil.BytesForBits(int64(input.Len()))))

	orig := input.NullBitmapBytes()
	if orig != nil {
		bitutil.CopyBitmap(orig, input.Data().Offset(), input.Len(), masked.Buffers()[0].Bytes(), 0)
	} else {
		bitutil.SetBitsTo(masked.Buffers()[0].Bytes(), 0, int64(input.Len()), true)
	}

	for _, i := range indices {
		bitutil.ClearBit(masked.Buffers()[0].Bytes(), i)
	}
	return array.MakeFromData(masked)
}

func TestCanCast(t *testing.T) {
	expectCanCast := func(from arrow.DataType, to []arrow.DataType, expected bool) {
		for _, dt := range to {
			assert.Equalf(t, expected, kernels.CanCast(from, dt), "\tFrom: %s\n\tTo: %s", from, dt)
		}
	}

	expectCanCast(arrow.Null, integerTypes, true)
	expectCanCast(arrow.FixedWidthTypes.Boolean, integerTypes, true)

	for _, fromNumeric := range integerTypes {
		// expectCanCast(fromNumeric, []arrow.DataType{arrow.FixedWidthTypes.Boolean}, true)
		expectCanCast(fromNumeric, integerTypes, true)
		expectCanCast(&arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: fromNumeric}, []arrow.DataType{fromNumeric}, true)
		expectCanCast(fromNumeric, []arrow.DataType{arrow.Null}, false)
	}
}

type CastTestSuite struct {
	suite.Suite

	registry functions.FunctionRegistry
	ectx     functions.ExecCtx

	castFn functions.Function
}

func (cs *CastTestSuite) SetupSuite() {
	cs.ectx.ChunkSize = math.MaxInt64
	kernels.RegisterScalarCasts(&cs.registry)
	var err error
	cs.castFn, err = cs.registry.GetFunction("cast")
	cs.Require().NoError(err)
	cs.Require().NotNil(cs.castFn)
}

func (cs *CastTestSuite) SetupTest() {
	cs.ectx.Mem = memory.NewCheckedAllocator(memory.NewGoAllocator())
	cs.ectx.Registry = &cs.registry
}

func (cs *CastTestSuite) TearDownTest() {
	cs.ectx.Mem.(*memory.CheckedAllocator).AssertSize(cs.T(), 0)
}

func (cs *CastTestSuite) assertBuffersSame(left, right arrow.ArrayData, buf int) {
	cs.Same(left.Buffers()[buf], right.Buffers()[buf])
}

func (cs *CastTestSuite) checkCast(input, expected arrow.Array, options compute.CastOptions) {
	options.ToType = expected.DataType()
	ctx := functions.SetExecCtx(context.Background(), &cs.ectx)

	in := compute.NewDatum(input)
	defer in.Release()
	out := compute.NewDatum(expected)
	defer out.Release()

	checkScalarUnary(cs.T(), ctx, cs.castFn, in, out, &options)
}

func (cs *CastTestSuite) checkCastFails(input arrow.Array, options compute.CastOptions) {
	ctx := functions.SetExecCtx(context.Background(), &cs.ectx)

	in := compute.NewDatum(input)
	defer in.Release()

	_, err := functions.ExecuteFunction(ctx, cs.castFn, []compute.Datum{in}, &options)
	cs.Error(err)
}

func (cs *CastTestSuite) checkCastZeroCopy(input arrow.Array, toType arrow.DataType, options compute.CastOptions) {
	ctx := functions.SetExecCtx(context.Background(), &cs.ectx)
	in := compute.NewDatum(input)
	defer in.Release()

	options.ToType = toType

	out, err := functions.ExecuteFunction(ctx, cs.castFn, []compute.Datum{in}, &options)
	cs.NoError(err)
	defer out.Release()

	result := out.(*compute.ArrayDatum)
	cs.Len(result.Value.Buffers(), len(input.Data().Buffers()))
	for i := range input.Data().Buffers() {
		cs.assertBuffersSame(input.Data(), result.Value, i)
	}
}

func (cs *CastTestSuite) checkZeroCopy(from, to arrow.DataType, str string) {
	arr, _, _ := array.FromJSON(cs.ectx.Mem, from, strings.NewReader(str))
	defer arr.Release()

	cs.checkCastZeroCopy(arr, to, *compute.DefaultCastOptions(true))
}

func (cs *CastTestSuite) TestSameTypeZeroCopy() {
	ctx := functions.SetExecCtx(context.Background(), &cs.ectx)
	arr, _, err := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[0, null, 2, 3, 4]`))
	cs.NoError(err)
	defer arr.Release()

	args := []compute.Datum{compute.NewDatum(arr)}
	defer args[0].Release()

	opts := compute.NewCastOptions(arrow.PrimitiveTypes.Int32, true)
	out, err := functions.ExecuteFunction(ctx, cs.castFn, args, opts)
	cs.NoError(err)
	defer out.Release()

	result := out.(*compute.ArrayDatum)
	cs.assertBuffersSame(arr.Data(), result.Value, 0)
	cs.assertBuffersSame(arr.Data(), result.Value, 1)
}

func (cs *CastTestSuite) TestCastDoesNotProvideDefaultOpts() {
	ctx := functions.SetExecCtx(context.Background(), &cs.ectx)
	arr, _, err := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[0, null, 2, 3, 4]`))
	cs.NoError(err)
	defer arr.Release()

	args := []compute.Datum{compute.NewDatum(arr)}
	defer args[0].Release()
	_, err = functions.ExecuteFunction(ctx, cs.castFn, args, nil)
	cs.Error(err)
}

func (cs *CastTestSuite) TestFromBoolean() {
	arr, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Boolean, strings.NewReader(`[true, false, null, true, false, true, true, null, false, false, true]`))
	cs.NoError(err)
	defer arr.Release()

	out, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[1, 0, null, 1, 0, 1, 1, null, 0, 0, 1]`))
	defer out.Release()

	cs.checkCast(arr, out, compute.CastOptions{})
}

func (cs *CastTestSuite) TestToIntUpcast() {
	arr, _, err := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int8, strings.NewReader(`[0, null, 127, -1, 0]`))
	cs.NoError(err)
	defer arr.Release()

	out, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[0, null, 127, -1, 0]`))
	defer out.Release()

	cs.checkCast(arr, out, compute.CastOptions{})

	arr, _, err = array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint8, strings.NewReader(`[0, 100, 200, 255, 0]`))
	cs.NoError(err)
	defer arr.Release()

	out, _, _ = array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int16, strings.NewReader(`[0, 100, 200, 255, 0]`))
	defer out.Release()

	cs.checkCast(arr, out, compute.CastOptions{})
}

func (cs *CastTestSuite) TestOverflowInNullSlot() {
	first, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[0, 87654321, 2000, 1000, 0]`))
	defer first.Release()
	masked := maskArrayWithNullsAt(cs.ectx.Mem, first, []int{1})
	defer masked.Release()

	out, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int16, strings.NewReader(`[0, null, 2000, 1000, 0]`))
	defer out.Release()

	cs.checkCast(masked, out, *compute.DefaultCastOptions(true))
}

func (cs *CastTestSuite) TestIntDowncastSafe() {
	tests := []struct {
		name          string
		from          arrow.DataType
		fromData      string
		to            arrow.DataType
		expectSuccess bool
	}{
		{"int16 to uint8", arrow.PrimitiveTypes.Int16, `[0, null, 200, 1, 2]`, arrow.PrimitiveTypes.Uint8, true},
		{"int16 to uint8 overflow", arrow.PrimitiveTypes.Int16, `[0, null, 256, 0, 0]`, arrow.PrimitiveTypes.Uint8, false},
		{"int16 to uint8 underflow", arrow.PrimitiveTypes.Int16, `[0, null, -1, 0, 0]`, arrow.PrimitiveTypes.Uint8, false},
		{"int32 to int16", arrow.PrimitiveTypes.Int32, `[0, null, 2000, 1, 2]`, arrow.PrimitiveTypes.Int16, true},
		{"int32 to int16 overflow", arrow.PrimitiveTypes.Int32, `[0, null, 2000, 70000, 2]`, arrow.PrimitiveTypes.Int16, false},
		{"int32 to int16 underflow", arrow.PrimitiveTypes.Int32, `[0, null, 2000, -70000, 2]`, arrow.PrimitiveTypes.Int16, false},
		{"int32 to uint 8 underflow", arrow.PrimitiveTypes.Int32, `[0, null, 2000, -70000, 2]`, arrow.PrimitiveTypes.Uint8, false},
	}

	for _, tt := range tests {
		cs.Run(tt.name, func() {
			from, _, _ := array.FromJSON(cs.ectx.Mem, tt.from, strings.NewReader(tt.fromData))
			defer from.Release()

			if tt.expectSuccess {
				to, _, _ := array.FromJSON(cs.ectx.Mem, tt.to, strings.NewReader(tt.fromData))
				defer to.Release()

				cs.checkCast(from, to, *compute.DefaultCastOptions(true))
			} else {
				cs.checkCastFails(from, *compute.NewCastOptions(tt.to, true))
			}
		})
	}
}

func (cs *CastTestSuite) TestIntegerSignedToUnsigned() {
	arr, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[-2147483648, null, -1, 65535, 2147483647]`))
	defer arr.Release()
	// same width
	cs.checkCastFails(arr, *compute.NewCastOptions(arrow.PrimitiveTypes.Uint32, true))
	// wider
	cs.checkCastFails(arr, *compute.NewCastOptions(arrow.PrimitiveTypes.Uint64, true))
	// narrower
	cs.checkCastFails(arr, *compute.NewCastOptions(arrow.PrimitiveTypes.Uint16, true))

	opts := compute.CastOptions{AllowIntOverflow: true}

	exu32, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint32, strings.NewReader(`[2147483648, null, 4294967295, 65535, 2147483647]`))
	defer exu32.Release()
	cs.checkCast(arr, exu32, opts)

	exu64, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint64, strings.NewReader(`[18446744071562067968, null, 18446744073709551615, 65535, 2147483647]`), array.WithUseNumber())
	defer exu64.Release()
	cs.checkCast(arr, exu64, opts)

	exu16, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint16, strings.NewReader(`[0, null, 65535, 65535, 65535]`))
	defer exu16.Release()
	cs.checkCast(arr, exu16, opts)

	// fail with overflow instead of underflow
	arr, _, _ = array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[0, null, 0, 65536, 2147483647]`))
	defer arr.Release()
	cs.checkCastFails(arr, *compute.NewCastOptions(arrow.PrimitiveTypes.Uint16, true))

	exu16, _, _ = array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint16, strings.NewReader(`[0, null, 0, 0, 65535]`))
	defer exu16.Release()

	cs.checkCast(arr, exu16, opts)
}

func (cs *CastTestSuite) TestUnsignedToSigned() {
	u32s, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Uint32, strings.NewReader(`[4294967295, null, 0, 32768]`))
	defer u32s.Release()

	// same width
	cs.checkCastFails(u32s, *compute.NewCastOptions(arrow.PrimitiveTypes.Int32, true))
	// narrower
	cs.checkCastFails(u32s, *compute.NewCastOptions(arrow.PrimitiveTypes.Int16, true))
	slice := array.NewSlice(u32s, 1, int64(u32s.Len()))
	defer slice.Release()
	cs.checkCastFails(slice, *compute.NewCastOptions(arrow.PrimitiveTypes.Int16, true))

	opts := compute.CastOptions{AllowIntOverflow: true}
	exint32, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int32, strings.NewReader(`[-1, null, 0, 32768]`))
	defer exint32.Release()
	cs.checkCast(u32s, exint32, opts)

	exint64, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int64, strings.NewReader(`[4294967295, null, 0, 32768]`))
	defer exint64.Release()
	cs.checkCast(u32s, exint64, opts)

	exint16, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int16, strings.NewReader(`[-1, null, 0, -32768]`))
	defer exint16.Release()
	cs.checkCast(u32s, exint16, opts)
}

func (cs *CastTestSuite) check(from, to arrow.DataType, start, expected string, opts compute.CastOptions) {
	arr, _, _ := array.FromJSON(cs.ectx.Mem, from, strings.NewReader(start), array.WithUseNumber())
	defer arr.Release()
	ex, _, _ := array.FromJSON(cs.ectx.Mem, to, strings.NewReader(expected), array.WithUseNumber())
	defer ex.Release()
	cs.checkCast(arr, ex, opts)
}

func (cs *CastTestSuite) checkFail(from arrow.DataType, data string, opts compute.CastOptions) {
	arr, _, _ := array.FromJSON(cs.ectx.Mem, from, strings.NewReader(data), array.WithUseNumber())
	defer arr.Release()
	cs.checkCastFails(arr, opts)
}

func (cs *CastTestSuite) TestIntDowncastUnsafe() {
	opts := compute.CastOptions{AllowIntOverflow: true}

	// int16 to uint8, no overflow/underflow
	cs.check(arrow.PrimitiveTypes.Int16, arrow.PrimitiveTypes.Uint8, `[0, null, 200, 1, 2]`, `[0, null, 200, 1, 2]`, opts)
	// int16 to uint8 with overflow/underflow
	cs.check(arrow.PrimitiveTypes.Int16, arrow.PrimitiveTypes.Uint8, `[0, null, 256, 1, 2, -1]`, `[0, null, 0, 1, 2, 255]`, opts)
	// int32 to int16 no overflow/underflow
	cs.check(arrow.PrimitiveTypes.Int32, arrow.PrimitiveTypes.Int16, `[0, null, 2000, 1, 2, -1]`, `[0, null, 2000, 1, 2, -1]`, opts)
	// int32 to int16 with overflow/underflow
	cs.check(arrow.PrimitiveTypes.Int32, arrow.PrimitiveTypes.Int16, `[0, null, 2000, 70000, -70000]`, `[0, null, 2000, 4464, -4464]`, opts)
}

func (cs *CastTestSuite) TestFloatingToInt() {
	for _, ft := range []arrow.DataType{arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		for _, it := range []arrow.DataType{arrow.PrimitiveTypes.Int32, arrow.PrimitiveTypes.Int64} {
			// float to int no trunc
			cs.check(ft, it, `[1.0, null, 0.0, -1.0, 5.0]`, `[1, null, 0, -1, 5]`, compute.CastOptions{})
			// float to int truncate error
			opts := compute.NewCastOptions(it, true)
			cs.checkFail(ft, `[1.5, 0.0, null, 0.5, -1.5, 5.5]`, *opts)

			// float to int truncate allowed
			opts.AllowFloatTruncate = true
			cs.check(ft, it, `[1.5, 0.0, null, 0.5, -1.5, 5.5]`, `[1, 0, null, 0, -1, 5]`, *opts)
		}
	}
}

func (cs *CastTestSuite) TestIntToFloating() {
	for _, from := range []arrow.DataType{arrow.PrimitiveTypes.Uint32, arrow.PrimitiveTypes.Int32} {
		two24 := `[16777216, 16777217]`
		arr, _, _ := array.FromJSON(cs.ectx.Mem, from, strings.NewReader(two24))
		defer arr.Release()

		cs.checkCastFails(arr, *compute.NewCastOptions(arrow.PrimitiveTypes.Float32, true))
		sl := array.NewSlice(arr, 0, 1)
		defer sl.Release()

		ex, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Float32, strings.NewReader(two24))
		defer ex.Release()
		sl2 := array.NewSlice(ex, 0, 1)
		defer sl2.Release()
		cs.checkCast(sl, sl2, *compute.NewCastOptions(arrow.PrimitiveTypes.Float32, true))
	}

	i64, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int64, strings.NewReader(`[-9223372036854775808, -9223372036854775807, 0, 9223372036854775806, 9223372036854775807]`), array.WithUseNumber())
	defer i64.Release()

	cs.checkCastFails(i64, *compute.NewCastOptions(arrow.PrimitiveTypes.Float64, true))

	// masking values with nulls makes it safe
	masked := maskArrayWithNullsAt(cs.ectx.Mem, i64, []int{0, 1, 3, 4})
	defer masked.Release()

	ex, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Float64, strings.NewReader(`[null, null, 0, null, null]`))
	defer ex.Release()

	cs.checkCast(masked, ex, compute.CastOptions{})

	cs.checkFail(arrow.PrimitiveTypes.Uint64, `[9007199254740992, 9007199254740993]`, *compute.NewCastOptions(arrow.PrimitiveTypes.Float64, true))
}

func (cs *CastTestSuite) TestDecimal128ToInt() {
	options := compute.NewCastOptions(arrow.PrimitiveTypes.Int64, true)

	for _, allowOverflow := range []bool{false, true} {
		for _, allowTruncate := range []bool{false, true} {
			options.AllowIntOverflow = allowOverflow
			options.AllowDecimalTruncate = allowTruncate

			// no overflow no truncate
			cs.check(&arrow.Decimal128Type{Precision: 38, Scale: 10}, arrow.PrimitiveTypes.Int64,
				`["02.0000000000", "-11.0000000000", "22.0000000000", "-121.0000000000", null]`,
				`[2, -11, 22, -121, null]`, *options)
		}
	}

	for _, allowOverflow := range []bool{false, true} {
		options.AllowIntOverflow = allowOverflow
		arr, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: 10},
			strings.NewReader(`["02.1000000000", "-11.0000004500", "22.0000004500", "-121.1210000000", null]`))
		defer arr.Release()
		ex, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int64, strings.NewReader(`[2, -11, 22, -121, null]`))
		defer ex.Release()
		options.AllowDecimalTruncate = true
		cs.checkCast(arr, ex, *options)

		options.AllowDecimalTruncate = false
		cs.checkCastFails(arr, *options)
	}

	for _, allowTruncate := range []bool{false, true} {
		options.AllowDecimalTruncate = allowTruncate
		arr, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: 10},
			strings.NewReader(`["12345678901234567890000.0000000000", "99999999999999999999999.0000000000", null]`))
		defer arr.Release()

		options.AllowIntOverflow = true
		ex, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int64,
			// 12345678901234567890000 % 2**64, 99999999999999999999999 % 2**64
			strings.NewReader(`[4807115922877858896, 200376420520689663, null]`), array.WithUseNumber())
		defer ex.Release()
		cs.checkCast(arr, ex, *options)
		options.AllowIntOverflow = false
		cs.checkCastFails(arr, *options)
	}

	for _, allowIntOverflow := range []bool{false, true} {
		for _, allowDecimalTruncate := range []bool{false, true} {
			options.AllowIntOverflow, options.AllowDecimalTruncate = allowIntOverflow, allowDecimalTruncate
			if options.AllowIntOverflow && options.AllowDecimalTruncate {
				cs.check(&arrow.Decimal128Type{Precision: 38, Scale: 10}, arrow.PrimitiveTypes.Int64,
					`["12345678901234567890000.0045345000", "99999999999999999999999.0000344300", null]`,
					// 12345678901234567890000 % 2**64, 99999999999999999999999 % 2**64
					`[4807115922877858896, 200376420520689663, null]`, *options)
			} else {
				cs.checkFail(&arrow.Decimal128Type{Precision: 38, Scale: 10},
					`["12345678901234567890000.0045345000", "99999999999999999999999.0000344300", null]`,
					*options)
			}
		}
	}

	bldr := array.NewDecimal128Builder(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: -4})
	defer bldr.Release()
	for _, d := range []decimal128.Num{decimal128.FromU64(1234567890000), decimal128.FromI64(-120000)} {
		var err error
		d, err = d.Rescale(0, -4)
		cs.NoError(err)
		bldr.Append(d)
	}
	negScale := bldr.NewArray()
	defer negScale.Release()

	options.AllowIntOverflow = true
	options.AllowDecimalTruncate = true

	ex, _, _ := array.FromJSON(cs.ectx.Mem, arrow.PrimitiveTypes.Int64, strings.NewReader(`[1234567890000, -120000]`))
	defer ex.Release()
	cs.checkCast(negScale, ex, *options)
}

func (cs *CastTestSuite) TestIntegerToDecimal() {
	dt := &arrow.Decimal128Type{Precision: 22, Scale: 2}
	for _, it := range integerTypes {
		cs.Run(it.Name(), func() {
			cs.check(it, dt, `[0, 7, null, 100, 99]`, `["0.00", "7.00", null, "100.00", "99.00"]`,
				*compute.DefaultCastOptions(true))
		})
	}

	// extreme values!
	cs.check(arrow.PrimitiveTypes.Int64, &arrow.Decimal128Type{Precision: 19, Scale: 0},
		`[-9223372036854775808, 9223372036854775807]`, `["-9223372036854775808", "9223372036854775807"]`,
		*compute.DefaultCastOptions(true))
	cs.check(arrow.PrimitiveTypes.Uint64, &arrow.Decimal128Type{Precision: 20, Scale: 0},
		`[0, 18446744073709551615]`, `["0", "18446744073709551615"]`, *compute.DefaultCastOptions(true))

	// insufficient output precision
	cs.checkFail(arrow.PrimitiveTypes.Int8, `[0]`, *compute.NewCastOptions(&arrow.Decimal128Type{Precision: 5, Scale: 3}, true))
}

func (cs *CastTestSuite) TestDecimal128ToDecimal128() {
	options := compute.DefaultCastOptions(true)
	cs.Run("basic", func() {
		for _, allowtrunc := range []bool{false, true} {
			cs.Run(strconv.FormatBool(allowtrunc), func() {
				options.AllowDecimalTruncate = allowtrunc
				noTrunc, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: 10},
					strings.NewReader(`["02.0000000000", "30.0000000000", "22.0000000000", "-121.0000000000", null]`))
				defer noTrunc.Release()

				ex, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 28, Scale: 0},
					strings.NewReader(`["02.", "30.", "22.", "-121.", null]`))
				defer ex.Release()

				cs.checkCast(noTrunc, ex, *options)
				cs.checkCast(ex, noTrunc, *options)
			})
		}
	})

	cs.Run("same scale, different precision", func() {
		for _, allowTrunc := range []bool{false, true} {
			options.AllowDecimalTruncate = allowTrunc
			cs.Run(strconv.FormatBool(allowTrunc), func() {
				d52, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 5, Scale: 2},
					strings.NewReader(`["12.34", "0.56"]`))
				defer d52.Release()
				d42, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 4, Scale: 2},
					strings.NewReader(`["12.34", "0.56"]`))
				defer d42.Release()

				cs.checkCast(d52, d42, *options)
				cs.checkCast(d42, d52, *options)
			})
		}
	})

	cs.Run("rescale trunc", func() {
		d38, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: 10},
			strings.NewReader(`["-02.1234567890", "30.1234567890", null]`))
		defer d38.Release()
		d28, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 28, Scale: 0},
			strings.NewReader(`["-02.", "30.", null]`))
		defer d28.Release()
		d38Roundtripped, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.Decimal128Type{Precision: 38, Scale: 10},
			strings.NewReader(`["-02.0000000000", "30.0000000000", null]`))
		defer d38Roundtripped.Release()

		options.AllowDecimalTruncate = true
		cs.checkCast(d38, d28, *options)
		cs.checkCast(d28, d38Roundtripped, *options)

		options.AllowDecimalTruncate = false
		options.ToType = d28.DataType()
		cs.checkCastFails(d38, *options)
		cs.checkCast(d28, d38Roundtripped, *options)
	})

	cs.Run("loss precision trunc", func() {
		d42Type := &arrow.Decimal128Type{Precision: 4, Scale: 2}
		d42Data := `["12.34"]`
		testpairs := []struct {
			dt   arrow.DataType
			data string
		}{
			{&arrow.Decimal128Type{Precision: 3, Scale: 2}, `["12.34"]`},
			{&arrow.Decimal128Type{Precision: 4, Scale: 3}, `["12.340"]`},
			{&arrow.Decimal128Type{Precision: 2, Scale: 1}, `["12.3"]`},
		}

		for _, tt := range testpairs {
			options.AllowDecimalTruncate = false
			options.ToType = tt.dt
			cs.checkFail(d42Type, d42Data, *options)
		}
	})
}

func (cs *CastTestSuite) TestFloatingToDecimal128() {
	for _, flt := range []arrow.DataType{arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		dt := &arrow.Decimal128Type{Precision: 5, Scale: 2}
		cs.check(flt, dt, `[0.0, null, 123.45, 123.456, 999.994]`, `["0.00", null, "123.45", "123.46", "999.99"]`, *compute.DefaultCastOptions(true))

		// overflow
		opts := compute.NewCastOptions(dt, true)
		cs.checkFail(flt, `[999.996]`, *opts)

		opts.AllowDecimalTruncate = true
		cs.check(flt, dt, `[0.0, null, 999.996, 123.45, 999.994]`, `["0.00", null, "0.00", "123.45", "999.99"]`, *opts)
	}

	// 2**64 + 2**41 (exactly representable as a float)
	cs.check(arrow.PrimitiveTypes.Float32, &arrow.Decimal128Type{Precision: 20, Scale: 0},
		`[1.8446746e+19, -1.8446746e+19]`, `["18446746272732807168", "-18446746272732807168"]`, *compute.DefaultCastOptions(true))

	cs.check(arrow.PrimitiveTypes.Float64, &arrow.Decimal128Type{Precision: 20, Scale: 0},
		`[1.8446744073709556e+19, -1.8446744073709556e+19]`, `["18446744073709555712", "-18446744073709555712"]`, *compute.DefaultCastOptions(true))

	cs.check(arrow.PrimitiveTypes.Float32, &arrow.Decimal128Type{Precision: 20, Scale: 4},
		`[1.8446746e+15, -1.8446746e+15]`, `["1844674627273280.7168", "-1844674627273280.7168"]`, *compute.DefaultCastOptions(true))

	cs.check(arrow.PrimitiveTypes.Float64, &arrow.Decimal128Type{Precision: 20, Scale: 4},
		`[1.8446744073709556e+15, -1.8446744073709556e+15]`, `["1844674407370955.5712", "-1844674407370955.5712"]`, *compute.DefaultCastOptions(true))

	// other edge cases tested with decimal128.FromFloat*
}

func (cs *CastTestSuite) TestDecimal128ToFloating() {
	for _, flt := range []arrow.DataType{arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		cs.check(&arrow.Decimal128Type{Precision: 5, Scale: 2}, flt,
			`["0.00", null, "123.45", "999.99"]`, `[0.0, null, 123.45, 999.99]`, *compute.DefaultCastOptions(true))
	}

	// other edge cases tested with decimal128.ToFloat*
}

const timestampJSON = `["1970-01-01T00:00:59.123456789","2000-02-29T23:23:23.999999999",
"1899-01-01T00:59:20.001001001","2033-05-18T03:33:20.000000000",
"2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
"2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004132",
"2010-01-01T05:25:25.005321", "2010-01-03T06:30:30.006163",
"2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
"2008-12-28", "2008-12-29", "2012-01-01 01:02:03", null]`

const timestampSecondsJSON = `["1970-01-01T00:00:59","2000-02-29T23:23:23",
"1899-01-01T00:59:20","2033-05-18T03:33:20",
"2020-01-01T01:05:05", "2019-12-31T02:10:10",
"2019-12-30T03:15:15", "2009-12-31T04:20:20",
"2010-01-01T05:25:25", "2010-01-03T06:30:30",
"2010-01-04T07:35:35", "2006-01-01T08:40:40",
"2005-12-31T09:45:45", "2008-12-28", "2008-12-29",
"2012-01-01 01:02:03", null]`

const timestampExremeJSON = `["1677-09-20T00:00:59.123456", "2262-04-13T23:23:23.999999"]`

func (cs *CastTestSuite) TestTimestampToDate() {
	arr, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_ns, strings.NewReader(timestampJSON))
	cs.Require().NoError(err)
	defer arr.Release()

	date32, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date32,
		strings.NewReader(`[0, 11016, -25932, 23148, 18262, 18261, 18260, 14609, 14610, 14612, 14613, 13149, 13148, 14241, 14242, 15340, null]`))
	cs.Require().NoError(err)
	defer date32.Release()

	date64, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64,
		strings.NewReader(`[0, 951782400000, -2240524800000, 1999987200000,
		1577836800000, 1577750400000, 1577664000000, 1262217600000,
		1262304000000, 1262476800000, 1262563200000, 1136073600000,
		1135987200000, 1230422400000, 1230508800000, 1325376000000, null]`))
	cs.Require().NoError(err)
	defer date64.Release()

	timestampExtreme, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_us, strings.NewReader(`["1677-09-20T00:00:59.123456", "2262-04-13T23:23:23.999999"]`))
	cs.Require().NoError(err)
	defer timestampExtreme.Release()

	date32Extreme, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date32, strings.NewReader(`[-106753, 106753]`))
	defer date32Extreme.Release()

	date64Extreme, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64, strings.NewReader(`[-9223459200000, 9223459200000]`))
	defer date64Extreme.Release()

	cs.checkCast(arr, date32, *compute.NewCastOptions(arrow.FixedWidthTypes.Date32, true))
	cs.checkCast(arr, date64, *compute.NewCastOptions(arrow.FixedWidthTypes.Date64, true))
	cs.checkCast(timestampExtreme, date32Extreme, *compute.NewCastOptions(arrow.FixedWidthTypes.Date32, true))
	cs.checkCast(timestampExtreme, date64Extreme, *compute.NewCastOptions(arrow.FixedWidthTypes.Date64, true))

	for _, u := range []arrow.TimeUnit{arrow.Second, arrow.Millisecond, arrow.Microsecond, arrow.Nanosecond} {
		cs.Run(u.String(), func() {
			dt := &arrow.TimestampType{Unit: u}
			arr, _, err := array.FromJSON(cs.ectx.Mem, dt, strings.NewReader(timestampSecondsJSON))
			cs.Require().NoError(err)
			defer arr.Release()

			cs.checkCast(arr, date32, *compute.DefaultCastOptions(true))
			cs.checkCast(arr, date64, *compute.DefaultCastOptions(true))
		})
	}
}

func (cs *CastTestSuite) TestZonedTimestampToDate() {
	timestamps, _, err := array.FromJSON(cs.ectx.Mem,
		&arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: "Pacific/Marquesas"},
		strings.NewReader(timestampJSON))
	cs.Require().NoError(err)
	defer timestamps.Release()

	date32, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date32,
		strings.NewReader(`[
			-1, 11016, -25933, 23147,
			18261, 18260, 18259, 14608,
			14609, 14611, 14612, 13148,
			13148, 14240, 14241, 15339, null
		]`))
	cs.Require().NoError(err)
	defer date32.Release()

	date64, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64,
		strings.NewReader(`[
			-86400000, 951782400000, -2240611200000, 1999900800000,
			1577750400000, 1577664000000, 1577577600000, 1262131200000,
			1262217600000, 1262390400000, 1262476800000, 1135987200000,
			1135987200000, 1230336000000, 1230422400000, 1325289600000, null
		]`))
	cs.Require().NoError(err)
	defer date64.Release()

	cs.checkCast(timestamps, date32, *compute.DefaultCastOptions(true))
	cs.checkCast(timestamps, date64, *compute.DefaultCastOptions(true))

	date32, _, err = array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date32,
		strings.NewReader(`[
			0, 11017, -25932, 23148,
			18262, 18261, 18260, 14609,
			14610, 14612, 14613, 13149,
			13148, 14241, 14242, 15340, null
		]`))
	cs.Require().NoError(err)
	defer date32.Release()

	date64, _, err = array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64,
		strings.NewReader(`[
			0, 951868800000, -2240524800000, 1999987200000, 1577836800000,
			1577750400000, 1577664000000, 1262217600000, 1262304000000,
			1262476800000, 1262563200000, 1136073600000, 1135987200000,
			1230422400000, 1230508800000, 1325376000000, null
		]`))
	cs.Require().NoError(err)
	defer date64.Release()

	for _, u := range []arrow.TimeUnit{arrow.Second, arrow.Millisecond, arrow.Microsecond, arrow.Nanosecond} {
		cs.Run("australia "+u.String(), func() {
			dt := &arrow.TimestampType{Unit: u, TimeZone: "Australia/Broken_Hill"}
			arr, _, err := array.FromJSON(cs.ectx.Mem, dt, strings.NewReader(timestampSecondsJSON))
			cs.Require().NoError(err)
			defer arr.Release()

			cs.checkCast(arr, date32, *compute.DefaultCastOptions(true))
			cs.checkCast(arr, date64, *compute.DefaultCastOptions(true))
		})
	}
	for _, u := range []arrow.TimeUnit{arrow.Second, arrow.Millisecond, arrow.Microsecond, arrow.Nanosecond} {
		cs.Run("invalid timezone "+u.String(), func() {
			dt := &arrow.TimestampType{Unit: u, TimeZone: "Mars/Mariner_Valley"}
			arr, _, err := array.FromJSON(cs.ectx.Mem, dt, strings.NewReader(timestampSecondsJSON))
			cs.Require().NoError(err)
			defer arr.Release()

			cs.checkCastFails(arr, *compute.NewCastOptions(arrow.FixedWidthTypes.Date32, false))
			cs.checkCastFails(arr, *compute.NewCastOptions(arrow.FixedWidthTypes.Date64, false))
		})
	}
}

func (cs *CastTestSuite) TestTimestampsToTime() {
	timestamps, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_ns, strings.NewReader(timestampJSON))
	defer timestamps.Release()

	timestampsExtreme, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_us, strings.NewReader(timestampExremeJSON))
	defer timestampsExtreme.Release()

	timestampsMicro, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_us,
		strings.NewReader(`[
			"1970-01-01T00:00:59.123456","2000-02-29T23:23:23.999999",
			"1899-01-01T00:59:20.001001","2033-05-18T03:33:20.000000",
			"2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
			"2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004132",
			"2010-01-01T05:25:25.005321", "2010-01-03T06:30:30.006163",
			"2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
			"2008-12-28", "2008-12-29", "2012-01-01 01:02:03", null]`))
	defer timestampsMicro.Release()

	timestampsMilli, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_ms,
		strings.NewReader(`[
			"1970-01-01T00:00:59.123","2000-02-29T23:23:23.999",
			"1899-01-01T00:59:20.001","2033-05-18T03:33:20.000",
			"2020-01-01T01:05:05.001", "2019-12-31T02:10:10.002",
			"2019-12-30T03:15:15.003", "2009-12-31T04:20:20.004",
			"2010-01-01T05:25:25.005", "2010-01-03T06:30:30.006",
			"2010-01-04T07:35:35", "2006-01-01T08:40:40", "2005-12-31T09:45:45",
			"2008-12-28", "2008-12-29", "2012-01-01 01:02:03", null]`))
	defer timestampsMilli.Release()

	timestampsSecond, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_s, strings.NewReader(timestampSecondsJSON))
	defer timestampsSecond.Release()

	times, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64ns,
		strings.NewReader(`[
			59123456789, 84203999999999, 3560001001001, 12800000000000,
			3905001000000, 7810002000000, 11715003000000, 15620004132000,
			19525005321000, 23430006163000, 27335000000000, 31240000000000,
			35145000000000, 0, 0, 3723000000000, null
		]`))
	cs.Require().NoError(err)
	defer times.Release()

	timesNanoMicro, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64us,
		strings.NewReader(`[
			59123456, 84203999999, 3560001001, 12800000000,
			3905001000, 7810002000, 11715003000, 15620004132,
			19525005321, 23430006163, 27335000000, 31240000000,
			35145000000, 0, 0, 3723000000, null
		]`))
	cs.Require().NoError(err)
	defer timesNanoMicro.Release()

	timesNanoMilli, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time32ms,
		strings.NewReader(`[
			59123, 84203999, 3560001, 12800000,
			3905001, 7810002, 11715003, 15620004,
			19525005, 23430006, 27335000, 31240000,
			35145000, 0, 0, 3723000, null
		]`))
	cs.Require().NoError(err)
	defer timesNanoMilli.Release()

	timesMicroNano, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64ns,
		strings.NewReader(`[
			59123456000, 84203999999000, 3560001001000, 12800000000000,
			3905001000000, 7810002000000, 11715003000000, 15620004132000,
			19525005321000, 23430006163000, 27335000000000, 31240000000000,
			35145000000000, 0, 0, 3723000000000, null
		]`))
	cs.Require().NoError(err)
	defer timesMicroNano.Release()

	timesMilliNano, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64ns,
		strings.NewReader(`[
			59123000000, 84203999000000, 3560001000000, 12800000000000,
			3905001000000, 7810002000000, 11715003000000, 15620004000000,
			19525005000000, 23430006000000, 27335000000000, 31240000000000,
			35145000000000, 0, 0, 3723000000000, null
		]`))
	cs.Require().NoError(err)
	defer timesMilliNano.Release()

	timesMilliMicro, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64us,
		strings.NewReader(`[
			59123000, 84203999000, 3560001000, 12800000000,
			3905001000, 7810002000, 11715003000, 15620004000,
			19525005000, 23430006000, 27335000000, 31240000000,
			35145000000, 0, 0, 3723000000, null
		]`))
	cs.Require().NoError(err)
	defer timesMilliMicro.Release()

	timesExtreme, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64us, strings.NewReader(`[59123456, 84203999999]`))
	cs.Require().NoError(err)
	defer timesExtreme.Release()

	timesSec, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time32s,
		strings.NewReader(`[
			59, 84203, 3560, 12800,
			3905, 7810, 11715, 15620,
			19525, 23430, 27335, 31240,
			35145, 0, 0, 3723, null
		]`))
	cs.Require().NoError(err)
	defer timesSec.Release()

	timesMilli, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time32ms,
		strings.NewReader(`[
			59000, 84203000, 3560000, 12800000,
			3905000, 7810000, 11715000, 15620000,
			19525000, 23430000, 27335000, 31240000,
			35145000, 0, 0, 3723000, null
		]`))
	cs.Require().NoError(err)
	defer timesMilli.Release()

	timesMicro, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64us,
		strings.NewReader(`[
			59000000, 84203000000, 3560000000, 12800000000,
			3905000000, 7810000000, 11715000000, 15620000000,
			19525000000, 23430000000, 27335000000, 31240000000,
			35145000000, 0, 0, 3723000000, null
		]`))
	cs.Require().NoError(err)
	defer timesMicro.Release()

	timesNano, _, err := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Time64ns,
		strings.NewReader(`[
			59000000000, 84203000000000, 3560000000000, 12800000000000,
			3905000000000, 7810000000000, 11715000000000, 15620000000000,
			19525000000000, 23430000000000, 27335000000000, 31240000000000,
			35145000000000, 0, 0, 3723000000000, null
		]`))
	cs.Require().NoError(err)
	defer timesNano.Release()

	cs.checkCast(timestamps, times, *compute.DefaultCastOptions(true))
	cs.checkCastFails(timestamps, *compute.NewCastOptions(arrow.FixedWidthTypes.Time64us, true))
	cs.checkCast(timestampsExtreme, timesExtreme, *compute.DefaultCastOptions(true))

	tssec, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_s, strings.NewReader(timestampSecondsJSON))
	defer tssec.Release()
	tsmilli, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_ms, strings.NewReader(timestampSecondsJSON))
	defer tsmilli.Release()
	tsmicro, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_us, strings.NewReader(timestampSecondsJSON))
	defer tsmicro.Release()
	tsnano, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Timestamp_ns, strings.NewReader(timestampSecondsJSON))
	defer tsnano.Release()

	cs.checkCast(tssec, timesSec, *compute.DefaultCastOptions(true))
	cs.checkCast(tssec, timesMilli, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmilli, timesSec, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmilli, timesMilli, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmicro, timesMicro, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmicro, timesNano, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmicro, timesMilli, *compute.DefaultCastOptions(true))
	cs.checkCast(tsmicro, timesSec, *compute.DefaultCastOptions(true))
	cs.checkCast(tsnano, timesNano, *compute.DefaultCastOptions(true))
	cs.checkCast(tsnano, timesMicro, *compute.DefaultCastOptions(true))
	cs.checkCast(tsnano, timesMilli, *compute.DefaultCastOptions(true))
	cs.checkCast(tsnano, timesSec, *compute.DefaultCastOptions(true))

	truncateOptions := compute.CastOptions{AllowTimeTruncate: true}

	// truncation tests
	cs.checkCastFails(timestamps, *compute.NewCastOptions(arrow.FixedWidthTypes.Time64us, true))
	cs.checkCastFails(timestamps, *compute.NewCastOptions(arrow.FixedWidthTypes.Time32ms, true))
	cs.checkCastFails(timestamps, *compute.NewCastOptions(arrow.FixedWidthTypes.Time32s, true))
	cs.checkCastFails(timestampsMicro, *compute.NewCastOptions(arrow.FixedWidthTypes.Time32ms, true))
	cs.checkCastFails(timestampsMicro, *compute.NewCastOptions(arrow.FixedWidthTypes.Time32s, true))
	cs.checkCastFails(timestampsMilli, *compute.NewCastOptions(arrow.FixedWidthTypes.Time32s, true))
	cs.checkCast(timestamps, timesNanoMicro, truncateOptions)
	cs.checkCast(timestamps, timesNanoMilli, truncateOptions)
	cs.checkCast(timestamps, timesSec, truncateOptions)
	cs.checkCast(timestampsMicro, timesNanoMilli, truncateOptions)
	cs.checkCast(timestampsMicro, timesSec, truncateOptions)
	cs.checkCast(timestampsMilli, timesSec, truncateOptions)

	// upscaling tests
	cs.checkCast(timestampsMicro, timesMicroNano, *compute.DefaultCastOptions(true))
	cs.checkCast(timestampsMilli, timesMilliNano, *compute.DefaultCastOptions(true))
	cs.checkCast(timestampsMilli, timesMilliMicro, *compute.DefaultCastOptions(true))
	cs.checkCast(timestampsSecond, timesNano, *compute.DefaultCastOptions(true))
	cs.checkCast(timestampsSecond, timesMicro, *compute.DefaultCastOptions(true))
	cs.checkCast(timestampsSecond, timesMilli, *compute.DefaultCastOptions(true))

	// invalid timezone
	for _, u := range []arrow.TimeUnit{arrow.Second, arrow.Millisecond, arrow.Microsecond, arrow.Nanosecond} {
		timestamps, _, _ := array.FromJSON(cs.ectx.Mem, &arrow.TimestampType{Unit: u, TimeZone: "Mars/Mariner_Valley"}, strings.NewReader(timestampSecondsJSON))
		defer timestamps.Release()
		if u == arrow.Second || u == arrow.Millisecond {
			cs.checkCastFails(timestamps, *compute.NewCastOptions(&arrow.Time32Type{Unit: u}, false))
		} else {
			cs.checkCastFails(timestamps, *compute.NewCastOptions(&arrow.Time64Type{Unit: u}, false))
		}
	}
}

func (cs *CastTestSuite) TestZonedTimestampToTime() {
	cs.check(&arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: "Pacific/Marquesas"}, arrow.FixedWidthTypes.Time64ns,
		timestampJSON, `[
			52259123456789, 50003999999999, 56480001001001, 65000000000000,
			56105001000000, 60010002000000, 63915003000000, 67820004132000,
			71725005321000, 75630006163000, 79535000000000, 83440000000000,
			945000000000, 52200000000000, 52200000000000, 55923000000000, null
		]`, *compute.DefaultCastOptions(true))

	const (
		timeSec = `[
			34259, 35603, 35960, 47000,
			41705, 45610, 49515, 53420,
			57325, 61230, 65135, 69040,
			72945, 37800, 37800, 41523, null
		]`
		timeMilli = `[
			34259000, 35603000, 35960000, 47000000,
			41705000, 45610000, 49515000, 53420000,
			57325000, 61230000, 65135000, 69040000,
			72945000, 37800000, 37800000, 41523000, null
		]`
		timeMicro = `[
			34259000000, 35603000000, 35960000000, 47000000000,
			41705000000, 45610000000, 49515000000, 53420000000,
			57325000000, 61230000000, 65135000000, 69040000000,
			72945000000, 37800000000, 37800000000, 41523000000, null
		]`
		timeNano = `[
			34259000000000, 35603000000000, 35960000000000, 47000000000000,
			41705000000000, 45610000000000, 49515000000000, 53420000000000,
			57325000000000, 61230000000000, 65135000000000, 69040000000000,
			72945000000000, 37800000000000, 37800000000000, 41523000000000, null
		]`
	)

	opts := *compute.DefaultCastOptions(true)
	cs.check(&arrow.TimestampType{Unit: arrow.Second, TimeZone: "Australia/Broken_Hill"}, arrow.FixedWidthTypes.Time32s,
		timestampSecondsJSON, timeSec, opts)
	cs.check(&arrow.TimestampType{Unit: arrow.Millisecond, TimeZone: "Australia/Broken_Hill"}, arrow.FixedWidthTypes.Time32ms,
		timestampSecondsJSON, timeMilli, opts)
	cs.check(&arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "Australia/Broken_Hill"}, arrow.FixedWidthTypes.Time64us,
		timestampSecondsJSON, timeMicro, opts)
	cs.check(&arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: "Australia/Broken_Hill"}, arrow.FixedWidthTypes.Time64ns,
		timestampSecondsJSON, timeNano, opts)
}

func (cs *CastTestSuite) TestTimeToTime() {
	type timeTypePair struct {
		coarse, fine arrow.DataType
	}

	options := compute.DefaultCastOptions(true)

	cases := []timeTypePair{
		{arrow.FixedWidthTypes.Time32s, arrow.FixedWidthTypes.Time32ms},
		{arrow.FixedWidthTypes.Time32ms, arrow.FixedWidthTypes.Time64us},
		{arrow.FixedWidthTypes.Time64us, arrow.FixedWidthTypes.Time64ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000, 1000, 2000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456, 1123, 2456]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}

	cases = []timeTypePair{
		{arrow.FixedWidthTypes.Time32s, arrow.FixedWidthTypes.Time64us},
		{arrow.FixedWidthTypes.Time32ms, arrow.FixedWidthTypes.Time64ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000000, 1000000, 2000000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456000, 1123000, 2456000]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}

	cases = []timeTypePair{
		{arrow.FixedWidthTypes.Time32s, arrow.FixedWidthTypes.Time64ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000000000, 1000000000, 2000000000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456000000, 1123000000, 2456000000]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}
}

func (cs *CastTestSuite) TestTimestampZeroCopy() {
	cs.checkZeroCopy(arrow.FixedWidthTypes.Timestamp_s, arrow.FixedWidthTypes.Timestamp_s, `[0, null, 2000, 1000, 0]`)
	cs.checkZeroCopy(arrow.FixedWidthTypes.Timestamp_s, arrow.PrimitiveTypes.Int64, `[0, null, 2000, 1000, 0]`)
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int64, arrow.FixedWidthTypes.Timestamp_s, `[0, null, 2000, 1000, 0]`)
}

func (cs *CastTestSuite) TestTimestampToTimestampMultiplyOverflow() {
	options := compute.NewCastOptions(arrow.FixedWidthTypes.Timestamp_ns, true)
	// 1000-01-01, 1800-01-01, 2000-01-01, 2300-01-01, 3000-01-01
	cs.checkFail(arrow.FixedWidthTypes.Timestamp_s, `[-30610224000, -5364662400, 946684800, 10413792000, 32503680000]`, *options)
}

func (cs *CastTestSuite) TestTimeZeroCopy() {
	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Time32s, arrow.PrimitiveTypes.Int32} {
		cs.checkZeroCopy(arrow.FixedWidthTypes.Time32s, dt, `[0, null, 2000, 1000, 0]`)
	}
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int32, arrow.FixedWidthTypes.Time32s, `[0, null, 2000, 1000, 0]`)

	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Time64us, arrow.PrimitiveTypes.Int64} {
		cs.checkZeroCopy(arrow.FixedWidthTypes.Time64us, dt, `[0, null, 2000, 1000, 0]`)
	}
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int64, arrow.FixedWidthTypes.Time64us, `[0, null, 2000, 1000, 0]`)
}

func (cs *CastTestSuite) TestDateToDate() {
	day32, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date32, strings.NewReader(`[0, null, 100, 1, 10]`))
	defer day32.Release()
	day64, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64,
		strings.NewReader(`[0, null, 8640000000, 86400000, 864000000]`))
	defer day64.Release()

	// multiply promotion
	cs.checkCast(day32, day64, *compute.DefaultCastOptions(true))
	// no truncation
	cs.checkCast(day64, day32, *compute.DefaultCastOptions(true))

	day64WillBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, arrow.FixedWidthTypes.Date64,
		strings.NewReader(`[0, null, 8640000123, 86400456, 864000789]`))
	defer day64WillBeTruncated.Release()

	options := compute.NewCastOptions(arrow.FixedWidthTypes.Date32, true)
	cs.checkCastFails(day64WillBeTruncated, *options)

	// divide truncate
	options.AllowTimeTruncate = true
	cs.checkCast(day64WillBeTruncated, day32, *options)
}

func (cs *CastTestSuite) TestDateZeroCopy() {
	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Date32, arrow.PrimitiveTypes.Int32} {
		cs.checkZeroCopy(arrow.FixedWidthTypes.Date32, dt, `[0, null, 2000, 1000, 0]`)
	}
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int32, arrow.FixedWidthTypes.Date32, `[0, null, 2000, 1000, 0]`)

	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Date64, arrow.PrimitiveTypes.Int64} {
		cs.checkZeroCopy(arrow.FixedWidthTypes.Date64, dt, `[0, null, 2000, 1000, 0]`)
	}
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int64, arrow.FixedWidthTypes.Date64, `[0, null, 2000, 1000, 0]`)
}

func (cs *CastTestSuite) TestDurationToDuration() {
	type durationTypePair struct {
		coarse, fine arrow.DataType
	}

	options := compute.DefaultCastOptions(true)

	cases := []durationTypePair{
		{arrow.FixedWidthTypes.Duration_s, arrow.FixedWidthTypes.Duration_ms},
		{arrow.FixedWidthTypes.Duration_ms, arrow.FixedWidthTypes.Duration_us},
		{arrow.FixedWidthTypes.Duration_us, arrow.FixedWidthTypes.Duration_ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000, 1000, 2000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456, 1123, 2456]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}

	cases = []durationTypePair{
		{arrow.FixedWidthTypes.Duration_s, arrow.FixedWidthTypes.Duration_us},
		{arrow.FixedWidthTypes.Duration_ms, arrow.FixedWidthTypes.Duration_ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000000, 1000000, 2000000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456000, 1123000, 2456000]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}

	cases = []durationTypePair{
		{arrow.FixedWidthTypes.Duration_s, arrow.FixedWidthTypes.Duration_ns},
	}
	for _, tp := range cases {
		coarse, _, _ := array.FromJSON(cs.ectx.Mem, tp.coarse, strings.NewReader(`[0, null, 200, 1, 2]`))
		defer coarse.Release()
		promoted, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200000000000, 1000000000, 2000000000]`))
		defer promoted.Release()

		cs.checkCast(coarse, promoted, *options)

		willBeTruncated, _, _ := array.FromJSON(cs.ectx.Mem, tp.fine, strings.NewReader(`[0, null, 200456000000, 1123000000, 2456000000]`))
		defer willBeTruncated.Release()

		options.AllowTimeTruncate = false
		options.ToType = tp.coarse
		cs.checkCastFails(willBeTruncated, *options)

		options.AllowTimeTruncate = true
		cs.checkCast(willBeTruncated, coarse, *options)
	}
}

func (cs *CastTestSuite) TestDurationZeroCopy() {
	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Duration_s, arrow.PrimitiveTypes.Int64} {
		cs.checkZeroCopy(arrow.FixedWidthTypes.Duration_s, dt, `[0, null, 2000, 1000, 0]`)
	}
	cs.checkZeroCopy(arrow.PrimitiveTypes.Int64, arrow.FixedWidthTypes.Duration_s, `[0, null, 2000, 1000, 0]`)
}

func (cs *CastTestSuite) TestMiscToFloating() {
	opts := *compute.DefaultCastOptions(true)
	for _, dt := range []arrow.DataType{arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		cs.check(arrow.PrimitiveTypes.Int16, dt, `[0, null, 200, 1, 2]`, `[0, null, 200, 1, 2]`, opts)
		cs.check(arrow.PrimitiveTypes.Float32, dt, `[0, null, 200, 1, 2]`, `[0, null, 200, 1, 2]`, opts)
		cs.check(arrow.FixedWidthTypes.Boolean, dt, `[true, null, false, false, true]`, `[1, null, 0, 0, 1]`, opts)
	}
}

func (cs *CastTestSuite) TestDurationToDurationMultiplyOverflow() {
	options := compute.NewCastOptions(arrow.FixedWidthTypes.Duration_ns, true)
	cs.checkFail(arrow.FixedWidthTypes.Duration_s, `[10000000000, 1, 2, 3, 10000000000]`, *options)
}

func (cs *CastTestSuite) TestIdentityCasts() {
	checkIdent := func(dt arrow.DataType, json string) {
		cs.checkZeroCopy(dt, dt, json)
	}

	checkIdent(arrow.Null, `[null, null, null]`)
	checkIdent(arrow.FixedWidthTypes.Boolean, `[false, true, null, false]`)
	for _, dt := range numericTypes {
		checkIdent(dt, `[1, 2, null, 4]`)
	}
	checkIdent(arrow.BinaryTypes.Binary, `["Rk9P", "YmFy"]`)
	checkIdent(arrow.BinaryTypes.String, `["foo", "bar"]`)
	checkIdent(&arrow.FixedSizeBinaryType{ByteWidth: 3}, `["Rk9P", "YmFy"]`)

	checkIdent(arrow.FixedWidthTypes.Time32ms, `[1, 2, 3, 4]`)
	checkIdent(arrow.FixedWidthTypes.Time64us, `[1, 2, 3, 4]`)
	checkIdent(arrow.FixedWidthTypes.Date32, `[1, 2, 3, 4]`)
	checkIdent(arrow.FixedWidthTypes.Date64, `[86400000, 0]`)
	checkIdent(arrow.FixedWidthTypes.Timestamp_s, `[1, 2, 3, 4]`)
}

func (cs *CastTestSuite) TestEmptyCast() {
	checkEmpty := func(from, to arrow.DataType) {
		in := array.MakeFromData(array.NewData(from, 0, []*memory.Buffer{nil, nil}, nil, 0, 0))
		out := array.MakeFromData(array.NewData(to, 0, []*memory.Buffer{nil, nil}, nil, 0, 0))

		cs.checkCast(in, out, *compute.DefaultCastOptions(true))
	}

	for _, dt := range numericTypes {
		checkEmpty(arrow.FixedWidthTypes.Boolean, dt)
		// checkEmpty(dt, arrow.FixedWidthTypes.Boolean)
	}
}

func TestCastFuncs(t *testing.T) {
	suite.Run(t, new(CastTestSuite))
}
