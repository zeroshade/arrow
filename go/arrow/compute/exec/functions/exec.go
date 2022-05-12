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

package functions

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/array"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/arrow/memory"
	"github.com/apache/arrow/go/v9/arrow/scalar"
	"github.com/apache/arrow/go/v9/internal/utils"
)

var (
	ErrNotImplemented = errors.New("not yet implemented")

	ebiPool  = sync.Pool{New: func() any { return &execBatchIterator{} }}
	execPool = sync.Pool{New: func() any {
		return &execSupport{
			output: make([]compute.Datum, 0)}
	}}
)

type execSupport struct {
	inputDescrs []compute.ValueDescr
	kctx        compute.KernelCtx
	initArgs    compute.KernelInitArgs
	sexec       scalarExecutor
	output      []compute.Datum
}

type execBatchIterator struct {
	args         []compute.Datum
	chunkIdxes   []int
	chunkPos     []int64
	pos, length  int64
	maxChunksize int64
}

func createExecBatchIterator(args []compute.Datum, maxChunksize int64) (*execBatchIterator, error) {
	length := int64(1)
	lengthSet := false
	for _, a := range args {
		switch arg := a.(type) {
		case compute.ArrayLikeDatum:
			if !lengthSet {
				length = arg.Len()
				lengthSet = true
			} else if length != arg.Len() {
				return nil, errors.New("array arguments must all be the same length")
			}
		case *compute.ScalarDatum:
			continue
		default:
			return nil, errors.New("execBatchIterator only works with Scalar Array and ChunkedArray args")
		}
	}

	maxChunksize = utils.Min(utils.Max(1, maxChunksize), length)
	ebi := ebiPool.Get().(*execBatchIterator)
	ebi.maxChunksize, ebi.args, ebi.length = maxChunksize, args, length
	if cap(ebi.chunkIdxes) < len(args) {
		ebi.chunkIdxes = make([]int, len(args))
	}
	ebi.chunkIdxes = ebi.chunkIdxes[:len(args)]
	if cap(ebi.chunkPos) < len(args) {
		ebi.chunkPos = make([]int64, len(args))
	}
	ebi.chunkPos = ebi.chunkPos[:len(args)]
	return ebi, nil
}

func putIteratorBack(ebi *execBatchIterator) {
	ebi.args = nil
	ebi.chunkIdxes = ebi.chunkIdxes[:0]
	ebi.chunkPos = ebi.chunkPos[:0]
	ebiPool.Put(ebi)
}

func (ebi *execBatchIterator) next(batch *compute.ExecBatch) bool {
	if ebi.pos == ebi.length {
		return false
	}

	itrsize := utils.Min(ebi.length-ebi.pos, ebi.maxChunksize)
	for i := 0; i < len(ebi.args) && itrsize > 0; i++ {
		if ebi.args[i].Kind() != compute.KindChunked {
			continue
		}

		arg := ebi.args[i].(*compute.ChunkedDatum).Value
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
		batch.Values = make([]compute.Datum, len(ebi.args))
	} else {
		batch.Values = batch.Values[:len(ebi.args)]
	}
	batch.Length = itrsize
	for i, a := range ebi.args {
		switch arg := a.(type) {
		case *compute.ScalarDatum:
			batch.Values[i] = compute.NewDatum(arg.Value)
		case *compute.ArrayDatum:
			if ebi.pos == 0 && itrsize == int64(arg.Value.Len()) {
				batch.Values[i] = arg
				continue
			}
			sliceData := array.NewSliceData(arg.Value, ebi.pos, itrsize+ebi.pos)
			batch.Values[i] = compute.NewDatum(sliceData)
			sliceData.Release()
		case *compute.ChunkedDatum:
			chunk := arg.Chunks()[ebi.chunkIdxes[i]]
			sliceData := array.NewSliceData(chunk.Data(), ebi.chunkPos[i], ebi.chunkPos[i]+itrsize)
			batch.Values[i] = compute.NewDatum(sliceData)
			sliceData.Release()
			ebi.chunkPos[i] += itrsize
		}
	}
	ebi.pos += itrsize
	debug.Assert(ebi.pos <= ebi.length, "execBatchIterator position is passed the length")
	return true
}

func checkOptions(fn compute.Function, opts compute.FunctionOptions) error {
	if opts == nil && fn.Doc().OptionsRequired {
		return fmt.Errorf("function '%s' cannot be called without options", fn.Name())
	}
	return nil
}

func checkAllValues(vals []compute.Datum) error {
	// value type is Array, Chunked, or Scalar
	// all of these are covered by the ArrayLikeDatum interface
	for _, v := range vals {
		if _, ok := v.(compute.ArrayLikeDatum); !ok {
			return fmt.Errorf("tried executing function with non-value type %s", v)
		}
	}
	return nil
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

func getNullGeneralized(datum compute.Datum) nullGeneralized {
	dtID := datum.Type().ID()
	switch {
	case dtID == arrow.NULL:
		return nullsAllNull
	case !hasValidityBitmap(dtID):
		return nullsAllValid
	case datum.Kind() == compute.KindScalar:
		if datum.(*compute.ScalarDatum).Value.IsValid() {
			return nullsAllValid
		}
		return nullsAllNull
	case datum.Kind() == compute.KindArray:
		arr := datum.(*compute.ArrayDatum)
		if arr.NullN() == 0 || arr.Value.Buffers()[0] == nil {
			return nullsAllValid
		}
		if arr.NullN() == arr.Len() {
			return nullsAllNull
		}
	}
	return nullsPerhaps
}

func allocDataBuffer(ctx *compute.KernelCtx, length int64, bitWidth int) *memory.Buffer {
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

var nullPropPool = sync.Pool{
	New: func() any { return &nullPropagator{} },
}

type nullPropagator struct {
	ctx            *compute.KernelCtx
	batch          *compute.ExecBatch
	arrsWithNulls  []arrow.ArrayData
	isAllNull      bool
	out            *array.Data
	bitmap         []byte
	preallocBitmap bool
}

func newNullPropagator(ctx *compute.KernelCtx, batch *compute.ExecBatch, out *array.Data) *nullPropagator {
	np := nullPropPool.Get().(*nullPropagator)
	np.ctx = ctx
	np.batch = batch
	np.out = out
	np.preallocBitmap = false

	for _, d := range batch.Values {
		ng := getNullGeneralized(d)
		if ng == nullsAllNull {
			np.isAllNull = true
		}
		if ng != nullsAllValid && d.Kind() == compute.KindArray {
			np.arrsWithNulls = append(np.arrsWithNulls, d.(*compute.ArrayDatum).Value)
		}
	}
	if np.out.Buffers()[0] != nil {
		np.preallocBitmap = true
		np.bitmap = np.out.Buffers()[0].Bytes()
	}
	return np
}

func putNullProp(np *nullPropagator) {
	np.ctx = nil
	np.batch = nil
	np.arrsWithNulls = np.arrsWithNulls[:0]
	np.bitmap = nil
	nullPropPool.Put(np)
}

func (np *nullPropagator) ensureAllocated() {
	if np.preallocBitmap {
		return
	}
	np.out.Buffers()[0] = np.ctx.AllocateBitmap(int64(np.out.Len()))
	np.bitmap = np.out.Buffers()[0].Bytes()
}

func (np *nullPropagator) allNullShortCircuit() {
	np.out.SetNullN(np.out.Len())
	if np.preallocBitmap {
		bitutil.SetBitsTo(np.bitmap, int64(np.out.Offset()), int64(np.out.Len()), false)
		return
	}

	// walk all the values with nulls instead of breaking on the first in case
	// we find a bitmap that can be reused in the non-preallocated case
	for _, arr := range np.arrsWithNulls {
		if arr.NullN() == arr.Len() && arr.Buffers()[0] != nil {
			if np.out.Buffers()[0] != nil {
				np.out.Buffers()[0].Release()
			}
			np.out.Buffers()[0] = arr.Buffers()[0]
			arr.Buffers()[0].Retain()
			return
		}
	}

	np.ensureAllocated()
	bitutil.SetBitsTo(np.bitmap, int64(np.out.Offset()), int64(np.out.Len()), false)
}

func (np *nullPropagator) propagateOne() {
	// one array
	arr := np.arrsWithNulls[0]
	arrBitmap := arr.Buffers()[0]
	np.out.SetNullN(arr.NullN())

	if np.preallocBitmap {
		bitutil.CopyBitmap(arrBitmap.Bytes(), arr.Offset(), arr.Len(), np.bitmap, np.out.Offset())
		return
	}

	// two cases when memory was not pre-allocated
	//
	// * Offset is zero: we reuse the bitmap as is
	// * Offset is nonzero but multiple of 8: we can slice the bitmap
	// * Offset is not a multiple of 8: must allocate and use CopyBitmap
	//
	// keep in mind that np.out.Offset is not permitted to be nonzero when
	// the bitmap is preallocated, and that precondition is checked above this
	// in the call stack.
	switch {
	case arr.Offset() == 0:
		if np.out.Buffers()[0] != nil {
			np.out.Buffers()[0].Release()
		}
		arrBitmap.Retain()
		np.out.Buffers()[0] = arrBitmap
	case bitutil.IsMultipleOf8(int64(arr.Offset())):
		if np.out.Buffers()[0] != nil {
			np.out.Buffers()[0].Release()
		}
		np.out.Buffers()[0] = memory.SliceBuffer(arrBitmap, arr.Offset()/8, int(bitutil.BytesForBits(int64(arr.Len()))))
	default:
		np.ensureAllocated()
		bitutil.CopyBitmap(arrBitmap.Bytes(), arr.Offset(), arr.Len(), np.bitmap, 0)
	}
}

func (np *nullPropagator) propagateMultiple() {
	// more than one array. bitmapAnd intersects their bitmaps
	// don't compute null count of intersection until we need it
	np.ensureAllocated()

	acc := func(left, right arrow.ArrayData) {
		debug.Assert(left.Buffers()[0] != nil, "nil buffer null propagation")
		debug.Assert(right.Buffers()[0] != nil, "nil buffer null propagation")
		bitutil.BitmapAnd(left.Buffers()[0].Bytes(), left.Offset(), right.Buffers()[0].Bytes(), right.Offset(),
			np.out.Buffers()[0].Bytes(), np.out.Offset(), np.out.Len())
	}

	debug.Assert(len(np.arrsWithNulls) > 1, "propagateMultiple should have more than 1 arrsWithnulls")
	acc(np.arrsWithNulls[0], np.arrsWithNulls[1])
	for _, arr := range np.arrsWithNulls[2:] {
		acc(np.out, arr)
	}
}

func (np *nullPropagator) execute() {
	if np.isAllNull {
		// all-null value (or scalar null or all-null array) we can short
		// circuit for speed
		np.allNullShortCircuit()
		return
	}

	// at this point, we know that all the values in np.arrsWithNulls are
	// arrays that are not all null. so there are a few cases:
	//
	// * No arrays: this is a no-op w/o preallocation but when the bitmap
	//   is preallocated we have to fill it with 1s
	// * One array, whose bitmap can be zero-copied (w/o preallocation, and
	//   when no byte is split) or copied (split byte or w/prealloc)
	// * More than one array, we must compute the intersection of the bitmaps
	//
	// But if the output offset is nonzero for some reason, we copy into output
	// unconditionally

	np.out.SetNullN(array.UnknownNullCount)
	switch len(np.arrsWithNulls) {
	case 0:
		// no arrays with nulls case
		np.out.SetNullN(0)
		if np.preallocBitmap {
			bitutil.SetBitsTo(np.bitmap, int64(np.out.Offset()), int64(np.out.Len()), true)
		}
	case 1:
		np.propagateOne()
	default:
		np.propagateMultiple()
	}
}

func propagateNulls(ctx *compute.KernelCtx, batch *compute.ExecBatch, out *array.Data) error {
	debug.Assert(out != nil, "propagateNulls output should not be nil")
	debug.Assert(len(out.Buffers()) > 0, "out buffers must already a slice with length > 0")

	if out.DataType().ID() == arrow.NULL {
		// null output type is a noop
		return nil
	}

	// this function is *ONLY* able to write into output with non-zero offset
	// when the bitmap is preallocated. this could be a debug.Assert, but for
	// now let's return an error
	if out.Offset() != 0 && out.Buffers()[0] == nil {
		return errors.New("can only propagate nulls into non-zero offset if bitmap is preallocated")
	}
	np := newNullPropagator(ctx, batch, out)
	np.execute()
	putNullProp(np)
	return nil
}

func haveChunkedArray(values []compute.Datum) bool {
	for _, v := range values {
		if v.Kind() == compute.KindChunked {
			return true
		}
	}
	return false
}

func toChunkedArray(values []compute.Datum, dt arrow.DataType) *arrow.Chunked {
	arrs := make([]arrow.Array, 0, len(values))
	for _, v := range values {
		defer v.Release()
		if v.Len() == 0 {
			continue
		}
		arr := v.(*compute.ArrayDatum).MakeArray()
		defer arr.Release()
		arrs = append(arrs, arr)
	}
	return arrow.NewChunked(dt, arrs)
}

var emptyExecCtx compute.ExecCtx

func ExecuteFunction(ctx context.Context, fn compute.Function, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
	if ef, ok := fn.(ExecutableFunc); ok {
		return ef.Execute(ctx, args, opts)
	}

	if opts == nil {
		if err := checkOptions(fn, opts); err != nil {
			return nil, err
		}
		opts = fn.DefaultOptions()
	}
	ectx := compute.GetExecCtx(ctx)
	if ectx == nil {
		ectx = &emptyExecCtx
	}
	return executeFunctionImpl(ectx, fn, args, opts)
}

func executeFunctionImpl(ectx *compute.ExecCtx, fn compute.Function, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
	if err := checkAllValues(args); err != nil {
		return nil, err
	}

	ep := execPool.Get().(*execSupport)
	defer func() {
		ep.inputDescrs = ep.inputDescrs[:0]
		ep.kctx.State = nil
		ep.output = ep.output[:0]
		execPool.Put(ep)
	}()

	if cap(ep.inputDescrs) < len(args) {
		ep.inputDescrs = make([]compute.ValueDescr, len(args))
	}

	inputDescrs := ep.inputDescrs[:len(args)]
	for i, a := range args {
		inputDescrs[i] = a.(compute.ArrayLikeDatum).Descr()
	}

	kernel, err := fn.DispatchExact(inputDescrs)
	if err != nil {
		return nil, err
	}

	// implicitly cast args! TODO
	ep.kctx.Ctx = ectx
	ep.initArgs.Kernel = kernel
	ep.initArgs.Inputs = inputDescrs
	ep.initArgs.Options = opts

	init := kernel.GetInit()
	if init != nil {
		ep.kctx.State, err = init(&ep.kctx, ep.initArgs)
		if err != nil {
			return nil, err
		}
	}

	var executor kernelExecutor
	switch fn.Kind() {
	case compute.FuncScalarKind:
		executor = &ep.sexec
	case compute.FuncVectorKind:
		return nil, errors.New("vector functions not implemented")
	case compute.FuncScalarAggKind:
		return nil, errors.New("scalar agg functions not implemented")
	case compute.FuncHashAggKind:
		return nil, errors.New("direct execution of hashagg functions not implemented")
	default:
		return nil, errors.New("invalid function type")
	}

	if err := executor.init(&ep.kctx, ep.initArgs); err != nil {
		return nil, err
	}

	ch := make(chan compute.Datum)
	done := make(chan bool)
	go func() {
		defer close(done)
		for d := range ch {
			ep.output = append(ep.output, d)
		}
	}()

	err = executor.execute(args, ch)
	close(ch)
	if err != nil {
		return nil, err
	}

	<-done
	final := executor.wrapResults(args, ep.output)
	return final, nil
}

type kernelExecutor interface {
	init(*compute.KernelCtx, compute.KernelInitArgs) error
	execute([]compute.Datum, chan<- compute.Datum) error
	wrapResults(args, outputs []compute.Datum) compute.Datum
	checkResultType(out compute.Datum, funcName string) error
}

type bufferPrealloc struct {
	bitWidth, addedLength int
}

type baseKernelExec struct {
	ctx              *compute.KernelCtx
	kernel           compute.Kernel
	outDescr         compute.ValueDescr
	outNumBuffers    int
	validityPrealloc bool
	dataPrealloc     []bufferPrealloc
	batchIterator    *execBatchIterator
}

func (b *baseKernelExec) execCtx() *compute.ExecCtx { return b.ctx.Ctx }

func (b *baseKernelExec) init(ctx *compute.KernelCtx, args compute.KernelInitArgs) (err error) {
	b.ctx = ctx
	b.kernel = args.Kernel
	b.outDescr, err = b.kernel.GetSignature().OutputType().Resolve(b.ctx, args.Inputs)
	if b.dataPrealloc == nil {
		b.dataPrealloc = []bufferPrealloc{}
	} else {
		b.dataPrealloc = b.dataPrealloc[:0]
	}
	return
}

func (b *baseKernelExec) checkResultType(out compute.Datum, funcName string) error {
	typ := out.Type()
	if typ != nil && !arrow.TypeEqual(typ, b.outDescr.Type) {
		return fmt.Errorf("kernel type result mismatch for function '%s': declared as %s, actual is %s",
			funcName, b.outDescr.Type, typ)
	}
	return nil
}

func (b *baseKernelExec) setupArgIteration(args []compute.Datum) (err error) {
	b.batchIterator, err = createExecBatchIterator(args, b.execCtx().ChunkSize)
	return
}

func (b *baseKernelExec) prepareOutput(length int) (arrow.ArrayData, error) {
	var (
		buffers = make([]*memory.Buffer, b.outNumBuffers)
		nulls   = array.UnknownNullCount
	)

	if b.validityPrealloc {
		buffers[0] = b.ctx.AllocateBitmap(int64(length))
		defer buffers[0].Release()
	}
	if b.kernel.GetNullHandling() == compute.NullOutputNotNull {
		nulls = 0
	}
	for i, dp := range b.dataPrealloc {
		if dp.bitWidth >= 0 {
			buffers[i+1] = allocDataBuffer(b.ctx, int64(length+dp.addedLength), dp.bitWidth)
			defer buffers[i+1].Release()
		}
	}
	return array.NewData(b.outDescr.Type, length, buffers, nil, nulls, 0), nil
}

type scalarExecutor struct {
	baseKernelExec

	preallocContiguous bool
	preallocated       arrow.ArrayData
}

func (se *scalarExecutor) wrapResults(inputs, outputs []compute.Datum) compute.Datum {
	if se.outDescr.Shape == compute.ShapeScalar {
		debug.Assert(len(outputs) == 1, "scalar output should have exactly one output")
		return outputs[0]
	}

	// if execution yielded multiple chunks (because large arrays were split
	// based on exec context params), then the result is a chunked array
	switch {
	case haveChunkedArray(inputs) || len(outputs) > 1:
		chnk := toChunkedArray(outputs, se.outDescr.Type)
		defer chnk.Release()
		return compute.NewDatum(chnk)
	case len(outputs) == 1:
		return outputs[0]
	default:
		// no outputs emitted, should we really return a 0-length array?
		arr := scalar.MakeArrayOfNull(se.outDescr.Type, 0, se.execCtx().Mem)
		defer arr.Release()
		return compute.NewDatum(arr)
	}
}

func (se *scalarExecutor) setupPrealloc(totalLen int64, args []compute.Datum) error {
	se.outNumBuffers = len(se.outDescr.Type.Layout().Buffers)
	outTypeID := se.outDescr.Type.ID()
	// default to no validity pre-allocation for following:
	//  - Output Array is nullarray
	//  - kernel_->null_handling is COMPUTE_NO_PREALLOC or OUTPUT_NOT_NULL
	se.validityPrealloc = false
	if outTypeID != arrow.NULL {
		if se.kernel.GetNullHandling() == compute.NullComputedPrealloc {
			se.validityPrealloc = true
		} else if se.kernel.GetNullHandling() == compute.NullIntersection {
			allInputValid := true
			for _, arg := range args {
				nullgen := getNullGeneralized(arg) == nullsAllValid
				allInputValid = allInputValid && nullgen
			}
			se.validityPrealloc = !allInputValid
		}
	}

	if se.kernel.GetMemAlloc() == compute.MemPrealloc {
		se.dataPrealloc = computeDataPrealloc(se.outDescr.Type, se.dataPrealloc)
	}

	allof := func(pre []bufferPrealloc) bool {
		for _, b := range pre {
			if b.bitWidth < 0 {
				return false
			}
		}
		return true
	}

	_, isnested := se.outDescr.Type.(arrow.NestedType)
	se.preallocContiguous =
		(se.execCtx().PreallocContiguous && se.kernel.CanWriteSlices() &&
			se.validityPrealloc && outTypeID != arrow.DICTIONARY && !isnested &&
			len(se.dataPrealloc) == se.outNumBuffers-1 && allof(se.dataPrealloc))

	var err error
	if se.preallocContiguous {
		se.preallocated, err = se.prepareOutput(int(totalLen))
	}
	return err
}

func (s *scalarExecutor) prepareExecute(args []compute.Datum) (err error) {
	if err := s.setupArgIteration(args); err != nil {
		return err
	}

	if s.outDescr.Shape == compute.ShapeArray {
		err = s.setupPrealloc(s.batchIterator.length, args)
	}
	return
}

func (s *scalarExecutor) prepareNextOutput(batch *compute.ExecBatch) (out compute.Datum, err error) {
	if s.outDescr.Shape == compute.ShapeArray {
		if s.preallocContiguous {
			batchStart := s.batchIterator.pos - batch.Length
			if batch.Length < s.batchIterator.length {
				data := array.NewSliceData(s.preallocated, batchStart, batchStart+batch.Length)
				defer data.Release()
				out = compute.NewDatum(data)
			} else {
				out = compute.NewDatum(s.preallocated)
			}
		} else {
			data, err := s.prepareOutput(int(batch.Length))
			if err != nil {
				return nil, err
			}
			defer data.Release()
			out = compute.NewDatum(data)
		}
	} else {
		out = compute.NewDatum(scalar.MakeNullScalar(s.outDescr.Type))
	}
	return
}

func (s *scalarExecutor) execute(args []compute.Datum, out chan<- compute.Datum) error {
	if err := s.prepareExecute(args); err != nil {
		return err
	}
	defer putIteratorBack(s.batchIterator)
	var batch compute.ExecBatch
	for s.batchIterator.next(&batch) {
		if err := s.executeBatch(&batch, out); err != nil {
			return err
		}
	}
	if s.preallocContiguous {
		out <- compute.NewDatum(s.preallocated)
	}
	return nil
}

func (s *scalarExecutor) executeBatch(batch *compute.ExecBatch, out chan<- compute.Datum) error {
	result, err := s.prepareNextOutput(batch)
	if err != nil {
		return err
	}
	defer result.Release()

	if s.outDescr.Shape == compute.ShapeArray {
		outArr := result.(*compute.ArrayDatum).Value.(*array.Data)
		if s.outDescr.Type.ID() == arrow.NULL {
			outArr.SetNullN(outArr.Len())
		} else if s.kernel.GetNullHandling() == compute.NullIntersection {
			if err := propagateNulls(s.ctx, batch, outArr); err != nil {
				return err
			}
		} else if s.kernel.GetNullHandling() == compute.NullOutputNotNull {
			outArr.SetNullN(0)
		}
	} else {
		if s.kernel.GetNullHandling() == compute.NullIntersection {
			// set scalar validity
			valid := true
			for _, v := range batch.Values {
				if !v.(*compute.ScalarDatum).Value.IsValid() {
					valid = false
					break
				}
			}
			result.(*compute.ScalarDatum).Value.SetValid(valid)
		} else if s.kernel.GetNullHandling() == compute.NullOutputNotNull {
			result.(*compute.ScalarDatum).Value.SetValid(true)
		}
	}

	if err := s.kernel.Execute(s.ctx, batch, result); err != nil {
		return err
	}

	if s.preallocContiguous {
		// some kernels simply nullify the validity bitmap when they
		// know there's 0 nulls. However this isn't compatible with
		// writing into slices
		if s.outDescr.Shape == compute.ShapeArray {
			debug.Assert(result.(*compute.ArrayDatum).Value.Buffers()[0] != nil, "null bitmap deleted by kernel but CanWriteIntoSlices = true")
		}
	} else {
		out <- compute.NewDatum(result)
	}
	return nil
}
