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

package bitutil

import (
	"math"
	"math/bits"
	"reflect"
	"unsafe"

	"github.com/apache/arrow/go/arrow/endian"
	"github.com/apache/arrow/go/arrow/internal/debug"
	"github.com/apache/arrow/go/arrow/memory"
)

var (
	BitMask        = [8]byte{1, 2, 4, 8, 16, 32, 64, 128}
	FlippedBitMask = [8]byte{254, 253, 251, 247, 239, 223, 191, 127}
)

// IsMultipleOf8 returns whether v is a multiple of 8.
func IsMultipleOf8(v int64) bool { return v&7 == 0 }

// IsMultipleOf64 returns whether v is a multiple of 64
func IsMultipleOf64(v int64) bool { return v&63 == 0 }

func BytesForBits(bits int64) int64 { return (bits + 7) >> 3 }

// NextPowerOf2 rounds x to the next power of two.
func NextPowerOf2(x int) int { return 1 << uint(bits.Len(uint(x))) }

// CeilByte rounds size to the next multiple of 8.
func CeilByte(size int) int { return (size + 7) &^ 7 }

// CeilByte64 rounds size to the next multiple of 8.
func CeilByte64(size int64) int64 { return (size + 7) &^ 7 }

// BitIsSet returns true if the bit at index i in buf is set (1).
func BitIsSet(buf []byte, i int) bool { return (buf[uint(i)/8] & BitMask[byte(i)%8]) != 0 }

// BitIsNotSet returns true if the bit at index i in buf is not set (0).
func BitIsNotSet(buf []byte, i int) bool { return (buf[uint(i)/8] & BitMask[byte(i)%8]) == 0 }

// SetBit sets the bit at index i in buf to 1.
func SetBit(buf []byte, i int) { buf[uint(i)/8] |= BitMask[byte(i)%8] }

// ClearBit sets the bit at index i in buf to 0.
func ClearBit(buf []byte, i int) { buf[uint(i)/8] &= FlippedBitMask[byte(i)%8] }

// SetBitTo sets the bit at index i in buf to val.
func SetBitTo(buf []byte, i int, val bool) {
	if val {
		SetBit(buf, i)
	} else {
		ClearBit(buf, i)
	}
}

// CountSetBits counts the number of 1's in buf up to n bits.
func CountSetBits(buf []byte, offset, n int) int {
	if offset > 0 {
		return countSetBitsWithOffset(buf, offset, n)
	}

	count := 0

	uint64Bytes := n / uint64SizeBits * 8
	for _, v := range bytesToUint64(buf[:uint64Bytes]) {
		count += bits.OnesCount64(v)
	}

	for _, v := range buf[uint64Bytes : n/8] {
		count += bits.OnesCount8(v)
	}

	// tail bits
	for i := n &^ 0x7; i < n; i++ {
		if BitIsSet(buf, i) {
			count++
		}
	}

	return count
}

func countSetBitsWithOffset(buf []byte, offset, n int) int {
	count := 0

	beg := offset
	end := offset + n

	begU8 := roundUp(beg, uint64SizeBits)

	init := min(n, begU8-beg)
	for i := offset; i < beg+init; i++ {
		if BitIsSet(buf, i) {
			count++
		}
	}

	nU64 := (n - init) / uint64SizeBits
	begU64 := begU8 / uint64SizeBits
	endU64 := begU64 + nU64
	bufU64 := bytesToUint64(buf)
	if begU64 < len(bufU64) {
		for _, v := range bufU64[begU64:endU64] {
			count += bits.OnesCount64(v)
		}
	}

	// FIXME: use a fallback to bits.OnesCount8
	// before counting the tail bits.

	tail := beg + init + nU64*uint64SizeBits
	for i := tail; i < end; i++ {
		if BitIsSet(buf, i) {
			count++
		}
	}

	return count
}

func roundUp(v, f int) int {
	return (v + (f - 1)) / f * f
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

const (
	uint64SizeBytes = int(unsafe.Sizeof(uint64(0)))
	uint64SizeBits  = uint64SizeBytes * 8
)

func bytesToUint64(b []byte) []uint64 {
	h := (*reflect.SliceHeader)(unsafe.Pointer(&b))

	var res []uint64
	s := (*reflect.SliceHeader)(unsafe.Pointer(&res))
	s.Data = h.Data
	s.Len = h.Len / uint64SizeBytes
	s.Cap = h.Cap / uint64SizeBytes

	return res
}

var (
	// PrecedingBitmask is a convenience set of values as bitmasks for checking
	// prefix bits of a byte
	PrecedingBitmask = [8]byte{0, 1, 3, 7, 15, 31, 63, 127}
	// TrailingBitmask is the bitwise complement version of kPrecedingBitmask
	TrailingBitmask = [8]byte{255, 254, 252, 248, 240, 224, 192, 128}
)

// SetBitsTo is a convenience function to quickly set or unset all the bits
// in a bitmap starting at startOffset for length bits.
func SetBitsTo(bits []byte, startOffset, length int64, areSet bool) {
	if length == 0 {
		return
	}

	beg := startOffset
	end := startOffset + length
	var fill uint8 = 0
	if areSet {
		fill = math.MaxUint8
	}

	byteBeg := beg / 8
	byteEnd := end/8 + 1

	// don't modify bits before the startOffset by using this mask
	firstByteMask := PrecedingBitmask[beg%8]
	// don't modify bits past the length by using this mask
	lastByteMask := TrailingBitmask[end%8]

	if byteEnd == byteBeg+1 {
		// set bits within a single byte
		onlyByteMask := firstByteMask
		if end%8 != 0 {
			onlyByteMask = firstByteMask | lastByteMask
		}

		bits[byteBeg] &= onlyByteMask
		bits[byteBeg] |= fill &^ onlyByteMask
		return
	}

	// set/clear trailing bits of first byte
	bits[byteBeg] &= firstByteMask
	bits[byteBeg] |= fill &^ firstByteMask

	if byteEnd-byteBeg > 2 {
		memory.Set(bits[byteBeg+1:byteEnd-1], fill)
	}

	if end%8 == 0 {
		return
	}

	bits[byteEnd-1] &= lastByteMask
	bits[byteEnd-1] |= fill &^ lastByteMask
}

var toFromLEFunc func(uint64) uint64

func init() {
	if endian.IsBigEndian {
		toFromLEFunc = bits.ReverseBytes64
	} else {
		toFromLEFunc = func(in uint64) uint64 { return in }
	}
}

type BitmapWriter struct {
	buf    []byte
	pos    int64
	length int64

	curByte    uint8
	bitMask    uint8
	byteOffset int64
}

// NewBitmapWriter returns a sequential bitwise writer that preserves surrounding
// bit values as it writes.
func NewBitmapWriter(bitmap []byte, start, length int64) *BitmapWriter {
	ret := &BitmapWriter{
		buf:        bitmap,
		length:     length,
		byteOffset: start / 8,
		bitMask:    BitMask[start%8],
	}
	if length > 0 {
		ret.curByte = bitmap[int(ret.byteOffset)]
	}
	return ret
}

func (b *BitmapWriter) Reset(start, length int64) {
	b.pos = 0
	b.byteOffset = start / 8
	b.bitMask = BitMask[start%8]
	b.length = length
	if b.length > 0 {
		b.curByte = b.buf[int(b.byteOffset)]
	}
}

func (b *BitmapWriter) Pos() int64 { return b.pos }
func (b *BitmapWriter) Set()       { b.curByte |= b.bitMask }
func (b *BitmapWriter) Clear()     { b.curByte &= ^b.bitMask }

func (b *BitmapWriter) Next() {
	b.bitMask = b.bitMask << 1
	b.pos++
	if b.bitMask == 0 {
		b.bitMask = 0x01
		b.buf[b.byteOffset] = b.curByte
		b.byteOffset++
		if b.pos < b.length {
			b.curByte = b.buf[int(b.byteOffset)]
		}
	}
}

func (b *BitmapWriter) Finish() {
	if b.length > 0 && (b.bitMask != 0x01 || b.pos < b.length) {
		b.buf[int(b.byteOffset)] = b.curByte
	}
}

type BitmapWordWriter struct {
	bitmap []byte
	offset int64
	len    int64

	bitMask     uint64
	currentWord uint64
}

func NewBitmapWordWriter(bitmap []byte, start int64, len int64) *BitmapWordWriter {
	ret := &BitmapWordWriter{
		bitmap:  bitmap[start/8:],
		len:     len,
		offset:  start % 8,
		bitMask: (uint64(1) << uint64(start%8)) - 1,
	}

	if ret.offset != 0 {
		if ret.len >= int64(unsafe.Sizeof(uint64(0))*8) {
			ret.currentWord = toFromLEFunc(endian.Native.Uint64(ret.bitmap))
		} else if ret.len > 0 {
			ret.currentWord = toFromLEFunc(uint64(ret.bitmap[0]))
		}
	}
	return ret
}

func (bm *BitmapWordWriter) PutNextWord(word uint64) {
	sz := int(unsafe.Sizeof(word))
	if bm.offset != 0 {
		// split one word into two adjacent words, don't touch unused bits
		//               |<------ word ----->|
		//               +-----+-------------+
		//               |  A  |      B      |
		//               +-----+-------------+
		//                  |         |
		//                  v         v       offset
		// +-------------+-----+-------------+-----+
		// |     ---     |  A  |      B      | --- |
		// +-------------+-----+-------------+-----+
		// |<------ next ----->|<---- current ---->|
		// fmt.Printf("%# b\n", word)
		word = (word << uint64(bm.offset)) | (word >> (int64(sz*8) - bm.offset))
		// fmt.Printf("%-#b\n", word)
		next := endian.Native.Uint64(bm.bitmap[sz:])
		// fmt.Printf("%#0b\n", bm.currentWord)
		bm.currentWord = (bm.currentWord & bm.bitMask) | (word &^ bm.bitMask)
		// fmt.Printf("%0#b\n", bm.currentWord)
		next = (next &^ bm.bitMask) | (word & bm.bitMask)
		// fmt.Printf("%#0b\n", next)
		endian.Native.PutUint64(bm.bitmap, toFromLEFunc(bm.currentWord))
		endian.Native.PutUint64(bm.bitmap[sz:], toFromLEFunc(next))
		bm.currentWord = next
	} else {
		endian.Native.PutUint64(bm.bitmap, toFromLEFunc(word))
	}
	bm.bitmap = bm.bitmap[sz:]
}

func (bm *BitmapWordWriter) PutNextTrailingByte(b byte, validBits int) {
	curbyte := (*[8]byte)(unsafe.Pointer(&bm.currentWord))[0]
	if validBits == 8 {
		if bm.offset != 0 {
			b = (b << bm.offset) | (b >> (8 - bm.offset))
			next := bm.bitmap[1]
			curbyte = (curbyte & byte(bm.bitMask)) | (b &^ byte(bm.bitMask))
			next = (next &^ byte(bm.bitMask)) | (b & byte(bm.bitMask))
			bm.bitmap[0] = curbyte
			bm.bitmap[1] = next
			bm.currentWord = uint64(next)
		} else {
			bm.bitmap[0] = b
		}
		bm.bitmap = bm.bitmap[1:]
	} else {
		debug.Assert(validBits > 0 && validBits < 8, "invalid valid bits in bitmap word writer")
		debug.Assert(BytesForBits(bm.offset+int64(validBits)) <= int64(len(bm.bitmap)), "writing trailiing byte outside of bounds of bitmap")
		wr := NewBitmapWriter(bm.bitmap, bm.offset, int64(validBits))
		for i := 0; i < validBits; i++ {
			if b&0x01 != 0 {
				wr.Set()
			} else {
				wr.Clear()
			}
			wr.Next()
			b >>= 1
		}
		wr.Finish()
	}
}

// BitmapReader is a simple bitmap reader for a byte slice.
type BitmapReader struct {
	bitmap []byte
	pos    int64
	len    int64

	current    byte
	byteOffset int64
	bitOffset  int64
}

// NewBitmapReader creates and returns a new bitmap reader for the given bitmap
func NewBitmapReader(bitmap []byte, offset, length int64) *BitmapReader {
	curbyte := byte(0)
	if length > 0 && bitmap != nil {
		curbyte = bitmap[offset/8]
	}
	return &BitmapReader{
		bitmap:     bitmap,
		byteOffset: offset / 8,
		bitOffset:  offset % 8,
		current:    curbyte,
		len:        length,
	}
}

// Set returns true if the current bit is set
func (b *BitmapReader) Set() bool {
	return (b.current & (1 << b.bitOffset)) != 0
}

// NotSet returns true if the current bit is not set
func (b *BitmapReader) NotSet() bool {
	return (b.current & (1 << b.bitOffset)) == 0
}

// Next advances the reader to the next bit in the bitmap.
func (b *BitmapReader) Next() {
	b.bitOffset++
	b.pos++
	if b.bitOffset == 8 {
		b.bitOffset = 0
		b.byteOffset++
		if b.pos < b.len {
			b.current = b.bitmap[int(b.byteOffset)]
		}
	}
}

// Pos returns the current bit position in the bitmap that the reader is looking at
func (b *BitmapReader) Pos() int64 { return b.pos }

// Len returns the total number of bits in the bitmap
func (b *BitmapReader) Len() int64 { return b.len }

type BitmapWordReader struct {
	bitmap        []byte
	offset        int64
	nwords        int64
	trailingBits  int
	trailingBytes int
	curword       uint64
}

func NewBitmapWordReader(bitmap []byte, offset, length int64) *BitmapWordReader {
	bitoffset := offset % 8
	byteOffset := offset / 8
	bm := &BitmapWordReader{
		offset: bitoffset,
		bitmap: bitmap[byteOffset : byteOffset+BytesForBits(bitoffset+length)],
		// decrement wordcount by 1 as we may touch two adjacent words in one iteration
		nwords: length/int64(unsafe.Sizeof(uint64(0))*8) - 1,
	}
	if bm.nwords < 0 {
		bm.nwords = 0
	}
	bm.trailingBits = int(length - bm.nwords*int64(unsafe.Sizeof(uint64(0)))*8)
	bm.trailingBytes = int(BytesForBits(int64(bm.trailingBits)))

	if bm.nwords > 0 {
		bm.curword = toFromLEFunc(endian.Native.Uint64(bm.bitmap))
	} else {
		bm.curword = toFromLEFunc(uint64(bm.bitmap[0]))
	}
	return bm
}

func (bm *BitmapWordReader) NextWord() uint64 {
	bm.bitmap = bm.bitmap[unsafe.Sizeof(bm.curword):]
	word := bm.curword
	nextWord := toFromLEFunc(endian.Native.Uint64(bm.bitmap))
	if bm.offset != 0 {
		// combine two adjacent words into one word
		// |<------ next ----->|<---- current ---->|
		// +-------------+-----+-------------+-----+
		// |     ---     |  A  |      B      | --- |
		// +-------------+-----+-------------+-----+
		//                  |         |       offset
		//                  v         v
		//               +-----+-------------+
		//               |  A  |      B      |
		//               +-----+-------------+
		//               |<------ word ----->|
		word >>= uint64(bm.offset)
		word |= nextWord << (int64(unsafe.Sizeof(uint64(0))*8) - bm.offset)
	}
	bm.curword = nextWord
	return word
}

func (bm *BitmapWordReader) NextTrailingByte() (val byte, validBits int) {
	debug.Assert(bm.trailingBits > 0, "next trailing byte called with no trailing bits")

	if bm.trailingBits <= 8 {
		// last byte
		validBits = bm.trailingBits
		bm.trailingBits = 0
		rdr := NewBitmapReader(bm.bitmap, bm.offset, int64(validBits))
		for i := 0; i < validBits; i++ {
			val >>= 1
			if rdr.Set() {
				val |= 0x80
			}
			rdr.Next()
		}
		val >>= (8 - validBits)
		return
	}

	// fmt.Printf("%#b\n", bm.curword)
	bm.bitmap = bm.bitmap[1:]
	nextByte := bm.bitmap[0]
	val = (*[8]byte)(unsafe.Pointer(&bm.curword))[0]
	if bm.offset != 0 {
		val >>= byte(bm.offset)
		val |= nextByte << (8 - bm.offset)
	}
	(*[8]byte)(unsafe.Pointer(&bm.curword))[0] = nextByte
	bm.trailingBits -= 8
	bm.trailingBytes--
	validBits = 8
	return
}

func (bm *BitmapWordReader) Words() int64       { return bm.nwords }
func (bm *BitmapWordReader) TrailingBytes() int { return bm.trailingBytes }

func CopyBitmap(src []byte, srcOffset, length int64, dst []byte, dstOffset int64) {
	if length == 0 {
		return
	}

	bitOffset := srcOffset % 8
	destBitOffset := dstOffset % 8

	if bitOffset != 0 || destBitOffset != 0 {
		rdr := NewBitmapWordReader(src, srcOffset, length)
		wr := NewBitmapWordWriter(dst, dstOffset, length)

		nwords := rdr.Words()
		for nwords > 0 {
			nwords--
			wr.PutNextWord(rdr.NextWord())
		}
		nbytes := rdr.TrailingBytes()
		for nbytes > 0 {
			nbytes--
			bt, validBits := rdr.NextTrailingByte()
			wr.PutNextTrailingByte(bt, validBits)
		}
		return
	}

	if length == 0 {
		return
	}

	nbytes := BytesForBits(length)

	// shift by its byte offset
	src = src[srcOffset/8:]
	dst = dst[dstOffset/8:]

	// Take care of the trailing bits in the last byte
	// E.g., if trailing_bits = 5, last byte should be
	// - low  3 bits: new bits from last byte of data buffer
	// - high 5 bits: old bits from last byte of dest buffer
	trailingBits := nbytes*8 - length
	trailMask := byte(uint(1)<<(8-trailingBits)) - 1

	copy(dst, src[:nbytes-1])
	lastData := src[nbytes-1]

	dst[nbytes-1] &= ^trailMask
	dst[nbytes-1] |= lastData & trailMask
}
