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
	"context"
	"fmt"

	"github.com/apache/arrow/go/v8/arrow"
	"github.com/apache/arrow/go/v8/arrow/array"
	"github.com/apache/arrow/go/v8/arrow/bitutil"
	"github.com/apache/arrow/go/v8/arrow/internal/debug"
	"github.com/apache/arrow/go/v8/arrow/memory"
	"github.com/apache/arrow/go/v8/arrow/scalar"
)

type KernelCtx struct {
	ctx   *ExecCtx
	state KernelState
}

func (k *KernelCtx) Allocate(nb int) *memory.Buffer {
	buf := memory.NewResizableBuffer(k.ctx.mem)
	buf.Resize(nb)
	return buf
}

func (k *KernelCtx) AllocateBitmap(nbits int64) *memory.Buffer {
	nbytes := bitutil.BytesForBits(nbits)
	return k.Allocate(int(nbytes))
}

type TypeMatcher interface {
	fmt.Stringer
	Matches(arrow.DataType) bool
	Equals(TypeMatcher) bool
}

type TypeKind int8

const (
	AnyType TypeKind = iota
	ExactType
	UseTypeMatcher
)

type InputType struct {
	kind        TypeKind
	shape       ValueShape
	dt          arrow.DataType
	typeMatcher TypeMatcher
}

func DefaultInputType() InputType {
	return InputType{kind: AnyType, shape: ShapeAny}
}

func NewAnyInput(shape ValueShape) InputType {
	return InputType{kind: AnyType, shape: shape}
}

func NewExactInput(dt arrow.DataType, shape ValueShape) InputType {
	return InputType{kind: ExactType, shape: shape, dt: dt}
}

func NewExactInputValue(descr ValueDescr) InputType {
	return NewExactInput(descr.Type, descr.Shape)
}

func NewInputMatcher(matcher TypeMatcher, shape ValueShape) InputType {
	return InputType{kind: UseTypeMatcher, shape: shape, typeMatcher: matcher}
}

type TypeResolver func(*KernelCtx, []ValueDescr) (ValueDescr, error)

type ResolveKind int8

const (
	ResolveFixed ResolveKind = iota
	ResolveComputed
)

type OutputType struct {
	kind     ResolveKind
	dt       arrow.DataType
	shape    ValueShape
	resolver TypeResolver
}

func NewOutputType(dt arrow.DataType) OutputType {
	return OutputType{dt: dt, kind: ResolveFixed}
}

func NewExactOutputValue(descr ValueDescr) OutputType {
	o := NewOutputType(descr.Type)
	o.shape = descr.Shape
	return o
}

func NewOutputTypeResolver(resolver TypeResolver) OutputType {
	return OutputType{kind: ResolveComputed, resolver: resolver}
}

func (o *OutputType) Resolve(ctx *KernelCtx, args []ValueDescr) (ValueDescr, error) {
	broadcasted := GetBroadcastShape(args)
	if o.kind == ResolveFixed {
		ret := ValueDescr{Type: o.dt, Shape: o.shape}
		if o.shape == ShapeAny {
			ret.Shape = broadcasted
		}
		return ret, nil
	}

	resolved, err := o.resolver(ctx, args)
	if err != nil {
		return ValueDescr{}, err
	}
	if resolved.Shape == ShapeAny {
		resolved.Shape = broadcasted
	}
	return resolved, nil
}

type KernelSig struct {
	inTypes []InputType
	outType OutputType
	varArgs bool

	cachedHash uint64
}

func newKernelSig(in []InputType, out OutputType, varargs bool) *KernelSig {
	debug.Assert(!varargs || len(in) >= 1, "must have multiple input types for kernel if varargs")
	return &KernelSig{
		inTypes: in,
		outType: out,
		varArgs: varargs,
	}
}

type NullHandling int8

const (
	NullIntersection NullHandling = iota
	NullComputedPrealloc
	NullComputeNoPrealloc
	NullOutputNotNull
)

type MemAlloc int8

const (
	MemPrealloc MemAlloc = iota
	MemNoPrealloc
)

type MinSIMDLevel int8

const (
	NoSIMD MinSIMDLevel = iota
	SSE4_2
	AVX
	AVX2
	AVX512
	NEON
	MAXSIMD
)

type KernelInitArgs struct {
	Kernel  Kernel
	Inputs  []ValueDescr
	Options FunctionOptions
}

type KernelState interface{}

type KernelInit func(context.Context, KernelInitArgs) (KernelState, error)

type kernel struct {
	Init           KernelInit
	Parallelizable bool
	SimdLevel      MinSIMDLevel
	Signature      *KernelSig
}

func (k *kernel) GetSignature() *KernelSig { return k.Signature }

func newKernel(inTypes []InputType, out OutputType, init KernelInit) kernel {
	return kernel{
		Init:           init,
		Signature:      newKernelSig(inTypes, out, false),
		Parallelizable: true,
	}
}

type ArrayKernelExec func(context.Context, ExecBatch, Datum) error

type ScalarKernel struct {
	kernel

	Exec               ArrayKernelExec
	CanWriteIntoSlices bool
	NullHandling       NullHandling
	MemAlloc           MemAlloc
}

func (s *ScalarKernel) GetNullHandling() NullHandling {
	return s.NullHandling
}

func (s *ScalarKernel) GetMemAlloc() MemAlloc { return s.MemAlloc }

func (s *ScalarKernel) CanWriteSlices() bool { return s.CanWriteIntoSlices }

func NewScalarKernel(in []InputType, out OutputType, exec ArrayKernelExec, init KernelInit) ScalarKernel {
	return ScalarKernel{
		kernel:             newKernel(in, out, init),
		Exec:               exec,
		CanWriteIntoSlices: true,
		NullHandling:       NullIntersection,
		MemAlloc:           MemPrealloc,
	}
}

func NewScalarKernelWithSig(sig *KernelSig, exec ArrayKernelExec, init KernelInit) ScalarKernel {
	return ScalarKernel{
		kernel:             kernel{Signature: sig, Init: init, Parallelizable: true},
		Exec:               exec,
		CanWriteIntoSlices: true,
		NullHandling:       NullIntersection,
		MemAlloc:           MemPrealloc,
	}
}

type Kernel interface {
	GetSignature() *KernelSig
	GetNullHandling() NullHandling
	GetMemAlloc() MemAlloc
	CanWriteSlices() bool
}

type kernelExecutor interface {
	init(context.Context, KernelInitArgs) error
	execute([]Datum) error
	wrapResults(args, outputs []Datum) Datum
	checkResultType(out Datum, funcName string) error
}

type bufferPrealloc struct {
	bitWidth, addedLength int
}

type kernelCtxKey struct{}

func GetKernelCtx(ctx context.Context) *KernelCtx {
	if v, ok := ctx.Value(kernelCtxKey{}).(*KernelCtx); ok {
		return v
	}
	return nil
}

type baseKernelExec struct {
	ctx              *KernelCtx
	kernel           Kernel
	outDescr         ValueDescr
	outNumBuffers    int
	validityPrealloc bool
	dataPrealloc     []bufferPrealloc
	batchIterator    *execBatchIterator
}

func (b *baseKernelExec) execCtx() *ExecCtx { return b.ctx.ctx }

func (b *baseKernelExec) init(ctx context.Context, args KernelInitArgs) (err error) {
	b.ctx = GetKernelCtx(ctx)
	b.kernel = args.Kernel
	b.outDescr, err = b.kernel.GetSignature().outType.Resolve(b.ctx, args.Inputs)
	b.dataPrealloc = []bufferPrealloc{}
	return
}

func (b *baseKernelExec) checkResultType(out Datum, funcName string) error {
	typ := out.Type()
	if typ != nil && !arrow.TypeEqual(typ, b.outDescr.Type) {
		return fmt.Errorf("kernel type result mismatch for function '%s': declared as %s, actual is %s",
			funcName, b.outDescr.Type, typ)
	}
	return nil
}

func (b *baseKernelExec) setupArgIteration(args []Datum) (err error) {
	b.batchIterator, err = createExecBatchIterator(args, b.execCtx().chunkSize)
	return
}

func (b *baseKernelExec) prepareOutput(length int) (arrow.ArrayData, error) {
	var (
		buffers = make([]*memory.Buffer, b.outNumBuffers)
		nulls   = array.UnknownNullCount
	)

	if b.validityPrealloc {
		buffers[0] = b.ctx.AllocateBitmap(int64(length))
	}
	if b.kernel.GetNullHandling() == NullOutputNotNull {
		nulls = 0
	}
	for i, dp := range b.dataPrealloc {
		if dp.bitWidth >= 0 {
			buffers[i+1] = allocDataBuffer(b.ctx, int64(length+dp.addedLength), dp.bitWidth)
		}
	}
	return array.NewData(b.outDescr.Type, length, buffers, nil, nulls, 0), nil
}

type scalarExecutor struct {
	baseKernelExec

	preallocContiguous bool
	preallocated       arrow.ArrayData
}

func (se *scalarExecutor) setupPrealloc(totalLen int64, args []Datum) error {
	se.outNumBuffers = len(se.outDescr.Type.Layout().Buffers)
	outTypeID := se.outDescr.Type.ID()
	// default to no validity pre-allocation for following:
	//  - Output Array is nullarray
	//  - kernel_->null_handling is COMPUTE_NO_PREALLOC or OUTPUT_NOT_NULL
	se.validityPrealloc = false
	if outTypeID != arrow.NULL {
		if se.kernel.GetNullHandling() == NullComputedPrealloc {
			se.validityPrealloc = true
		} else if se.kernel.GetNullHandling() == NullIntersection {
			allInputValid := true
			for _, arg := range args {
				nullgen := getNullGeneralized(arg) == nullsAllValid
				allInputValid = allInputValid && nullgen
			}
			se.validityPrealloc = !allInputValid
		}
	}

	if se.kernel.GetMemAlloc() == MemPrealloc {
		computeDataPrealloc(se.outDescr.Type, se.dataPrealloc)
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
		(se.execCtx().preallocContiguous && se.kernel.CanWriteSlices() &&
			se.validityPrealloc && outTypeID != arrow.DICTIONARY && !isnested &&
			len(se.dataPrealloc) == se.outNumBuffers-1 && allof(se.dataPrealloc))

	var err error
	if se.preallocContiguous {
		se.preallocated, err = se.prepareOutput(int(totalLen))
	}
	return err
}

func (s *scalarExecutor) prepareExecute(args []Datum) (err error) {
	if err := s.setupArgIteration(args); err != nil {
		return err
	}

	if s.outDescr.Shape == ShapeArray {
		err = s.setupPrealloc(s.batchIterator.length, args)
	}
	return
}

func (s *scalarExecutor) prepareNextOutput(batch *ExecBatch, out Datum) error {
	defer out.Release()
	if s.outDescr.Shape == ShapeArray {
		if s.preallocContiguous {
			batchStart := s.batchIterator.pos - batch.Length
			if batch.Length < s.batchIterator.length {
				data := array.NewSliceData(s.preallocated, batchStart, batchStart+batch.Length)
				defer data.Release()
				out = NewDatum(data)
			} else {
				out = NewDatum(s.preallocated)
			}
		} else {
			data, err := s.prepareOutput(int(batch.Length))
			if err != nil {
				return err
			}
			defer data.Release()
			out = NewDatum(data)
		}
	} else {
		out = NewDatum(scalar.MakeNullScalar(s.outDescr.Type))
	}
	return nil
}

func (s *scalarExecutor) execute(args []Datum, out chan<- Datum) error {
	if err := s.prepareExecute(args); err != nil {
		return err
	}
	var batch ExecBatch
	for s.batchIterator.next(&batch) {

	}
}

func (s *scalarExecutor) executeBatch(batch *ExecBatch, out chan<- Datum) error {

}
