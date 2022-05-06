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
		// opts := compute.NewCastOptions(dt, true)
		// cs.checkFail(flt, `[999.996]`, *opts)

		// opts.AllowDecimalTruncate = true
		// cs.check(flt, dt, `[0.0, null, 999.996, 123.45, 999.994]`, `["0.00", null, "0.00", "123.45", "999.99"]`, *opts)
	}
}

func TestCastFuncs(t *testing.T) {
	suite.Run(t, new(CastTestSuite))
}
