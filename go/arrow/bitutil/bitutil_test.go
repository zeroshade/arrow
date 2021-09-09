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

package bitutil_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"

	"github.com/apache/arrow/go/arrow/bitutil"
	"github.com/apache/arrow/go/arrow/internal/testing/tools"
	"github.com/stretchr/testify/assert"
)

func TestIsMultipleOf8(t *testing.T) {
	for _, tc := range []struct {
		v    int64
		want bool
	}{
		{-16, true},
		{-9, false},
		{-8, true},
		{-7, false},
		{-4, false},
		{-1, false},
		{-0, true},
		{0, true},
		{1, false},
		{4, false},
		{7, false},
		{8, true},
		{9, false},
		{16, true},
	} {
		t.Run(fmt.Sprintf("v=%d", tc.v), func(t *testing.T) {
			got := bitutil.IsMultipleOf8(tc.v)
			if got != tc.want {
				t.Fatalf("IsMultipleOf8(%d): got=%v, want=%v", tc.v, got, tc.want)
			}
		})
	}
}

func TestCeilByte(t *testing.T) {
	tests := []struct {
		name    string
		in, exp int
	}{
		{"zero", 0, 0},
		{"five", 5, 8},
		{"sixteen", 16, 16},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := bitutil.CeilByte(test.in)
			assert.Equal(t, test.exp, got)
		})
	}
}

func TestBitIsSet(t *testing.T) {
	buf := make([]byte, 2)
	buf[0] = 0xa1
	buf[1] = 0xc2
	exp := []bool{true, false, false, false, false, true, false, true, false, true, false, false, false, false, true, true}
	var got []bool
	for i := 0; i < 0x10; i++ {
		got = append(got, bitutil.BitIsSet(buf, i))
	}
	assert.Equal(t, exp, got)
}

func TestBitIsNotSet(t *testing.T) {
	buf := make([]byte, 2)
	buf[0] = 0xa1
	buf[1] = 0xc2
	exp := []bool{false, true, true, true, true, false, true, false, true, false, true, true, true, true, false, false}
	var got []bool
	for i := 0; i < 0x10; i++ {
		got = append(got, bitutil.BitIsNotSet(buf, i))
	}
	assert.Equal(t, exp, got)
}

func TestClearBit(t *testing.T) {
	buf := make([]byte, 2)
	buf[0] = 0xff
	buf[1] = 0xff
	for i, v := range []bool{false, true, true, true, true, false, true, false, true, false, true, true, true, true, false, false} {
		if v {
			bitutil.ClearBit(buf, i)
		}
	}
	assert.Equal(t, []byte{0xa1, 0xc2}, buf)
}

func TestSetBit(t *testing.T) {
	buf := make([]byte, 2)
	for i, v := range []bool{true, false, false, false, false, true, false, true, false, true, false, false, false, false, true, true} {
		if v {
			bitutil.SetBit(buf, i)
		}
	}
	assert.Equal(t, []byte{0xa1, 0xc2}, buf)
}

func TestSetBitTo(t *testing.T) {
	buf := make([]byte, 2)
	for i, v := range []bool{true, false, false, false, false, true, false, true, false, true, false, false, false, false, true, true} {
		bitutil.SetBitTo(buf, i, v)
	}
	assert.Equal(t, []byte{0xa1, 0xc2}, buf)
}

func TestSetBitsTo(t *testing.T) {
	for _, fillByte := range []byte{0x00, 0xFF} {
		{
			// set within a byte
			bm := []byte{fillByte, fillByte, fillByte, fillByte}
			bitutil.SetBitsTo(bm, 2, 2, true)
			bitutil.SetBitsTo(bm, 4, 2, false)
			assert.Equal(t, []byte{(fillByte &^ 0x3C) | 0xC}, bm[:1])
		}
		{
			// test straddling a single byte boundary
			bm := []byte{fillByte, fillByte, fillByte, fillByte}
			bitutil.SetBitsTo(bm, 4, 7, true)
			bitutil.SetBitsTo(bm, 11, 7, false)
			assert.Equal(t, []byte{(fillByte & 0xF) | 0xF0, 0x7, fillByte &^ 0x3}, bm[:3])
		}
		{
			// test byte aligned end
			bm := []byte{fillByte, fillByte, fillByte, fillByte}
			bitutil.SetBitsTo(bm, 4, 4, true)
			bitutil.SetBitsTo(bm, 8, 8, false)
			assert.Equal(t, []byte{(fillByte & 0xF) | 0xF0, 0x00, fillByte}, bm[:3])
		}
		{
			// test byte aligned end, multiple bytes
			bm := []byte{fillByte, fillByte, fillByte, fillByte}
			bitutil.SetBitsTo(bm, 0, 24, false)
			falseByte := byte(0)
			assert.Equal(t, []byte{falseByte, falseByte, falseByte, fillByte}, bm)
		}
	}
}

func TestCountSetBits(t *testing.T) {
	tests := []struct {
		name string
		buf  []byte
		off  int
		n    int
		exp  int
	}{
		{"some 03 bits", bbits(0x11000000), 0, 3, 2},
		{"some 11 bits", bbits(0x11000011, 0x01000000), 0, 11, 5},
		{"some 72 bits", bbits(0x11001010, 0x11110000, 0x00001111, 0x11000011, 0x11001010, 0x11110000, 0x00001111, 0x11000011, 0x10001001), 0, 9 * 8, 35},
		{"all  08 bits", bbits(0x11111110), 0, 8, 7},
		{"all  03 bits", bbits(0x11100001), 0, 3, 3},
		{"all  11 bits", bbits(0x11111111, 0x11111111), 0, 11, 11},
		{"all  72 bits", bbits(0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111), 0, 9 * 8, 72},
		{"none 03 bits", bbits(0x00000001), 0, 3, 0},
		{"none 11 bits", bbits(0x00000000, 0x00000000), 0, 11, 0},
		{"none 72 bits", bbits(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000), 0, 9 * 8, 0},

		{"some 03 bits - offset+1", bbits(0x11000000), 1, 3, 1},
		{"some 03 bits - offset+2", bbits(0x11000000), 2, 3, 0},
		{"some 11 bits - offset+1", bbits(0x11000011, 0x01000000, 0x00000000), 1, 11, 4},
		{"some 11 bits - offset+2", bbits(0x11000011, 0x01000000, 0x00000000), 2, 11, 3},
		{"some 11 bits - offset+3", bbits(0x11000011, 0x01000000, 0x00000000), 3, 11, 3},
		{"some 11 bits - offset+6", bbits(0x11000011, 0x01000000, 0x00000000), 6, 11, 3},
		{"some 11 bits - offset+7", bbits(0x11000011, 0x01000000, 0x00000000), 7, 11, 2},
		{"some 11 bits - offset+8", bbits(0x11000011, 0x01000000, 0x00000000), 8, 11, 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := bitutil.CountSetBits(test.buf, test.off, test.n)
			assert.Equal(t, test.exp, got)
		})
	}
}

func TestCountSetBitsOffset(t *testing.T) {
	slowCountSetBits := func(buf []byte, offset, n int) int {
		count := 0
		for i := offset; i < offset+n; i++ {
			if bitutil.BitIsSet(buf, i) {
				count++
			}
		}
		return count
	}

	const (
		bufSize = 1000
		nbits   = bufSize * 8
	)

	offsets := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 37, 63, 64, 128, nbits - 30, nbits - 64}

	buf := make([]byte, bufSize)

	rng := rand.New(rand.NewSource(0))
	_, err := rng.Read(buf)
	if err != nil {
		t.Fatal(err)
	}

	for i, offset := range offsets {
		want := slowCountSetBits(buf, offset, nbits-offset)
		got := bitutil.CountSetBits(buf, offset, nbits-offset)
		if got != want {
			t.Errorf("offset[%2d/%2d]=%5d. got=%5d, want=%5d", i+1, len(offsets), offset, got, want)
		}
	}
}

func writeToWriter(vals []int, wr *bitutil.BitmapWriter) {
	for _, v := range vals {
		if v != 0 {
			wr.Set()
		} else {
			wr.Clear()
		}
		wr.Next()
	}
	wr.Finish()
}

func bitmapFromSlice(vals []int, offset int64) []byte {
	out := make([]byte, bitutil.BytesForBits(int64(len(vals))+offset))
	wr := bitutil.NewBitmapWriter(out, offset, int64(len(vals)))
	writeToWriter(vals, wr)
	return out
}

func TestBitmapWriter(t *testing.T) {
	for _, fillByte := range []byte{0x00, 0xFF} {
		{
			bitmap := []byte{fillByte, fillByte, fillByte, fillByte}
			wr := bitutil.NewBitmapWriter(bitmap, 0, 12)
			writeToWriter([]int{0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1}, wr)
			// {0b00110110, 0b....1010, ........, ........}
			assert.Equal(t, []byte{0x36, (0x0A | (fillByte & 0xF0)), fillByte, fillByte}, bitmap)
		}
		{
			bitmap := []byte{fillByte, fillByte, fillByte, fillByte}
			wr := bitutil.NewBitmapWriter(bitmap, 3, 12)
			writeToWriter([]int{0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1}, wr)
			// {0b10110..., 0b.1010001, ........, ........}
			assert.Equal(t, []byte{0xb0 | (fillByte & 0x07), 0x51 | (fillByte & 0x80), fillByte, fillByte}, bitmap)
		}
		{
			bitmap := []byte{fillByte, fillByte, fillByte, fillByte}
			wr := bitutil.NewBitmapWriter(bitmap, 20, 12)
			writeToWriter([]int{0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1}, wr)
			// {........, ........, 0b0110...., 0b10100011}
			assert.Equal(t, []byte{fillByte, fillByte, 0x60 | (fillByte & 0x0f), 0xa3}, bitmap)
		}
	}
}

func TestBitmapReader(t *testing.T) {
	assertReaderVals := func(vals []int, rdr *bitutil.BitmapReader) {
		for _, v := range vals {
			if v != 0 {
				assert.True(t, rdr.Set())
				assert.False(t, rdr.NotSet())
			} else {
				assert.False(t, rdr.Set())
				assert.True(t, rdr.NotSet())
			}
			rdr.Next()
		}
	}

	for _, offset := range []int64{0, 1, 3, 5, 7, 8, 12, 13, 21, 38, 75, 120} {
		bm := bitmapFromSlice([]int{0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1}, offset)

		rdr := bitutil.NewBitmapReader(bm, offset, 14)
		assertReaderVals([]int{0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1}, rdr)
	}
}

func TestCopyBitmap(t *testing.T) {
	const bufsize = 18
	lengths := []int64{bufsize*8 - 4, bufsize * 8}
	offsets := []int64{0, 12, 16, 32, 37, 63, 64, 128}

	buffer := make([]byte, bufsize)

	// random bytes
	r := rand.New(rand.NewSource(0))
	r.Read(buffer)

	// add 16 byte padding
	otherBuffer := make([]byte, bufsize+32)
	r.Read(otherBuffer)

	for _, nbits := range lengths {
		for _, offset := range offsets {
			for _, destOffset := range offsets {
				t.Run(fmt.Sprintf("bits %d off %d dst %d", nbits, offset, destOffset), func(t *testing.T) {
					copyLen := nbits - offset

					bmCopy := make([]byte, len(otherBuffer))
					copy(bmCopy, otherBuffer)

					bitutil.CopyBitmap(buffer, offset, copyLen, bmCopy, destOffset)

					for i := 0; i < int(destOffset); i++ {
						assert.Equalf(t, bitutil.BitIsSet(otherBuffer, i), bitutil.BitIsSet(bmCopy, i), "bit index: %d", i)
					}
					for i := 0; i < int(copyLen); i++ {
						assert.Equalf(t, bitutil.BitIsSet(buffer, i+int(offset)), bitutil.BitIsSet(bmCopy, i+int(destOffset)), "bit index: %d", i)
					}
					for i := int(destOffset + copyLen); i < len(otherBuffer); i++ {
						assert.Equalf(t, bitutil.BitIsSet(otherBuffer, i), bitutil.BitIsSet(bmCopy, i), "bit index: %d", i)
					}
				})
			}
		}
	}
}

func bbits(v ...int32) []byte {
	return tools.IntsToBitsLSB(v...)
}

func BenchmarkBitIsSet(b *testing.B) {
	buf := make([]byte, 32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bitutil.BitIsSet(buf, (i%32)&0x1a)
	}
}

func BenchmarkSetBit(b *testing.B) {
	buf := make([]byte, 32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bitutil.SetBit(buf, (i%32)&0x1a)
	}
}

func BenchmarkSetBitTo(b *testing.B) {
	vals := []bool{true, false, false, false, false, true, false, true, false, true, false, false, false, false, true, true}
	buf := make([]byte, 32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bitutil.SetBitTo(buf, i%32, vals[i%len(vals)])
	}
}

var (
	intval int
)

func benchmarkCountSetBitsN(b *testing.B, offset, n int) {
	nn := n/8 + 1
	buf := make([]byte, nn)
	//src := [4]byte{0x1f, 0xaa, 0xba, 0x11}
	src := [4]byte{0x01, 0x01, 0x01, 0x01}
	for i := 0; i < nn; i++ {
		buf[i] = src[i&0x3]
	}
	b.ResetTimer()
	var res int
	for i := 0; i < b.N; i++ {
		res = bitutil.CountSetBits(buf, offset, n-offset)
	}
	intval = res
}

func BenchmarkCountSetBits_3(b *testing.B) {
	benchmarkCountSetBitsN(b, 0, 3)
}

func BenchmarkCountSetBits_32(b *testing.B) {
	benchmarkCountSetBitsN(b, 0, 32)
}

func BenchmarkCountSetBits_128(b *testing.B) {
	benchmarkCountSetBitsN(b, 0, 128)
}

func BenchmarkCountSetBits_1000(b *testing.B) {
	benchmarkCountSetBitsN(b, 0, 1000)
}

func BenchmarkCountSetBits_1024(b *testing.B) {
	benchmarkCountSetBitsN(b, 0, 1024)
}

func BenchmarkCountSetBitsOffset_3(b *testing.B) {
	benchmarkCountSetBitsN(b, 1, 3)
}

func BenchmarkCountSetBitsOffset_32(b *testing.B) {
	benchmarkCountSetBitsN(b, 1, 32)
}

func BenchmarkCountSetBitsOffset_128(b *testing.B) {
	benchmarkCountSetBitsN(b, 1, 128)
}

func BenchmarkCountSetBitsOffset_1000(b *testing.B) {
	benchmarkCountSetBitsN(b, 1, 1000)
}

func BenchmarkCountSetBitsOffset_1024(b *testing.B) {
	benchmarkCountSetBitsN(b, 1, 1024)
}

func benchmarkCopyBitmapN(b *testing.B, offsetSrc, offsetDest int64, n int) {
	nbits := int64(n * 8)
	// random bytes
	r := rand.New(rand.NewSource(0))
	src := make([]byte, n)
	r.Read(src)

	length := nbits - offsetSrc

	dest := make([]byte, bitutil.BytesForBits(length+offsetDest))

	b.ResetTimer()
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		bitutil.CopyBitmap(src, offsetSrc, length, dest, offsetDest)
	}
}

// Fast path which is just a memcopy
func BenchmarkCopyBitmapWithoutOffset(b *testing.B) {
	for _, sz := range []int{32, 128, 1000, 1024} {
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			benchmarkCopyBitmapN(b, 0, 0, sz)
		})
	}
}

// slow path where the source buffer is not byte aligned
func BenchmarkCopyBitmapWithOffset(b *testing.B) {
	for _, sz := range []int{32, 128, 1000, 1024} {
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			benchmarkCopyBitmapN(b, 4, 0, sz)
		})
	}
}

// slow path where both source and dest are not byte aligned
func BenchmarkCopyBitmapWithOffsetBoth(b *testing.B) {
	for _, sz := range []int{32, 128, 1000, 1024} {
		b.Run(strconv.Itoa(sz), func(b *testing.B) {
			benchmarkCopyBitmapN(b, 3, 7, sz)
		})
	}
}
