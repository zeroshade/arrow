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

package compute

import (
	"errors"
	"math"

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/array"
	"github.com/apache/arrow/go/v8/arrow/bitutil"
	"github.com/apache/arrow/go/v8/arrow/internal/debug"
	"github.com/apache/arrow/go/v8/arrow/memory"
	"github.com/apache/arrow/go/v8/internal/utils"
)

type SelectionVector struct {
	data    arrow.ArrayData
	indices []int32
}

type ExecBatch struct {
	Values          []Datum
	SelectionVector SelectionVector
	Guarantee       Expression
	Length          int64
}

type execBatchIterator struct {
	args         []Datum
	chunkIdxes   []int
	chunkPos     []int64
	pos, length  int64
	maxChunksize int64
}

func createExecBatchIterator(args []Datum, maxChunksize int64) (*execBatchIterator, error) {
	length := int64(1)
	lengthSet := false
	for _, a := range args {
		switch arg := a.(type) {
		case ArrayLikeDatum:
			if !lengthSet {
				length = arg.Len()
				lengthSet = true
			} else if length != arg.Len() {
				return nil, errors.New("array arguments must all be the same length")
			}
		case *ScalarDatum:
			continue
		default:
			return nil, errors.New("execBatchIterator only works with Scalar Array and ChunkedArray args")
		}
	}

	maxChunksize = utils.Min(maxChunksize, length)
	return &execBatchIterator{
		args:         args,
		length:       length,
		maxChunksize: maxChunksize,
		chunkIdxes:   make([]int, len(args)),
		chunkPos:     make([]int64, len(args)),
	}, nil
}

func (ebi *execBatchIterator) next(batch *ExecBatch) bool {
	if ebi.pos == ebi.length {
		return false
	}

	itrsize := utils.Min(ebi.length-ebi.pos, ebi.maxChunksize)
	for i := 0; i < len(ebi.args) && itrsize > 0; i++ {
		if ebi.args[i].Kind() != KindChunked {
			continue
		}

		arg := ebi.args[i].(*ChunkedDatum).Value
		var currentChunk arrow.Array
		for {
			currentChunk = arg.Chunk(ebi.chunkIdxes[i])
			if ebi.chunkPos[i] == int64(currentChunk.Len()) {
				ebi.chunkPos[i] = 0
				ebi.chunkIdxes[i]++
				continue
			}
			break
		}
		itrsize = utils.Min(int64(currentChunk.Len())-ebi.chunkPos[i], itrsize)
	}

	if cap(batch.Values) < len(ebi.args) {
		batch.Values = make([]Datum, len(ebi.args))
	} else {
		batch.Values = batch.Values[:len(ebi.args)]
	}
	batch.Length = itrsize
	for i, a := range ebi.args {
		switch arg := a.(type) {
		case *ScalarDatum:
			batch.Values[i] = NewDatum(arg.Value)
		case *ArrayDatum:
			sliceData := array.NewSliceData(arg.Value, ebi.pos, itrsize+ebi.pos)
			batch.Values[i] = NewDatum(sliceData)
			sliceData.Release()
		case *ChunkedDatum:
			chunk := arg.Chunks()[ebi.chunkIdxes[i]]
			sliceData := array.NewSliceData(chunk.Data(), ebi.chunkPos[i], ebi.chunkPos[i]+itrsize)
			batch.Values[i] = NewDatum(sliceData)
			sliceData.Release()
			ebi.chunkPos[i] += itrsize
		}
	}
	ebi.pos += itrsize
	debug.Assert(ebi.pos <= ebi.length, "execBatchIterator position is passed the length")
	return true
}

type ExecCtx struct {
	mem                memory.Allocator
	chunkSize          int64
	preallocContiguous bool
	useGoroutines      bool
}

func DefaultExecCtx() *ExecCtx {
	return &ExecCtx{
		mem:                memory.DefaultAllocator,
		chunkSize:          math.MaxInt64,
		preallocContiguous: true,
		useGoroutines:      true,
	}
}

func hasValidityBitmap(id arrow.Type) bool {
	switch id {
	case arrow.NULL, arrow.DENSE_UNION, arrow.SPARSE_UNION:
		return false
	}
	return true
}

type nullGeneralized int8

const (
	nullsPerhaps nullGeneralized = iota
	nullsAllValid
	nullsAllNull
)

func getNullGeneralized(datum Datum) nullGeneralized {
	dtID := datum.Type().ID()
	switch {
	case dtID == arrow.NULL:
		return nullsAllNull
	case !hasValidityBitmap(dtID):
		return nullsAllValid
	case datum.Kind() == KindScalar:
		if datum.(*ScalarDatum).Value.IsValid() {
			return nullsAllValid
		}
		return nullsAllNull
	case datum.Kind() == KindArray:
		arr := datum.(*ArrayDatum)
		if arr.NullN() == 0 || arr.Value.Buffers()[0] == nil {
			return nullsAllValid
		}
		if arr.NullN() == arr.Len() {
			return nullsAllNull
		}
	}
	return nullsPerhaps
}

func allocDataBuffer(ctx *KernelCtx, length int64, bitWidth int) *memory.Buffer {
	if bitWidth == 1 {
		return ctx.AllocateBitmap(length)
	}

	bufferSize := bitutil.BytesForBits(length * int64(bitWidth))
	return ctx.Allocate(int(bufferSize))
}

func computeDataPrealloc(dt arrow.DataType, widths []bufferPrealloc) []bufferPrealloc {
	if fixed, ok := dt.(arrow.FixedWidthDataType); ok && dt.ID() != arrow.NULL {
		return append(widths, bufferPrealloc{bitWidth: fixed.BitWidth()})
	}

	switch dt.ID() {
	case arrow.BINARY, arrow.STRING, arrow.LIST, arrow.MAP:
		return append(widths, bufferPrealloc{bitWidth: 32, addedLength: 1})
	}
	return widths
}
