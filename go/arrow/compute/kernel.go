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

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
	"github.com/apache/arrow/go/v9/arrow/memory"
	"github.com/apache/arrow/go/v9/internal/utils"
)

type KernelCtx struct {
	Ctx   *ExecCtx
	State KernelState
}

func (k *KernelCtx) Allocate(nb int) *memory.Buffer {
	buf := memory.NewResizableBuffer(k.Ctx.Mem)
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

type sameIDMatcher struct {
	id arrow.Type
}

func (s *sameIDMatcher) Matches(t arrow.DataType) bool { return s.id == t.ID() }
func (s *sameIDMatcher) String() string {
	return "Type::" + s.id.String()
}
func (s *sameIDMatcher) Equals(t TypeMatcher) bool {
	if s == t {
		return true
	}

	if m, ok := t.(*sameIDMatcher); ok {
		return s.id == m.id
	}
	return false
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

func NewInputIDType(id arrow.Type) InputType {
	return NewInputMatcher(&sameIDMatcher{id}, ShapeAny)
}

func (it *InputType) Kind() TypeKind { return it.kind }

func (it *InputType) Matches(descr ValueDescr) bool {
	if it.shape != ShapeAny && it.shape != descr.Shape {
		return false
	}

	switch it.kind {
	case ExactType:
		return arrow.TypeEqual(it.dt, descr.Type)
	case UseTypeMatcher:
		return it.typeMatcher.Matches(descr.Type)
	default:
		// ANY TYPE!
		return true
	}
}

func (it *InputType) MatchesDatum(val Datum) bool {
	return it.Matches(val.(ArrayLikeDatum).Descr())
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

func (o OutputType) Resolve(ctx *KernelCtx, args []ValueDescr) (ValueDescr, error) {
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

func NewKernelSig(in []InputType, out OutputType, varargs bool) *KernelSig {
	debug.Assert(!varargs || len(in) >= 1, "must have multiple input types for kernel if varargs")
	return &KernelSig{
		inTypes: in,
		outType: out,
		varArgs: varargs,
	}
}

func (k *KernelSig) VarArgs() bool { return k.varArgs }

func (k *KernelSig) OutputType() OutputType { return k.outType }

func (k *KernelSig) InputTypes() []InputType { return k.inTypes }

func (k *KernelSig) MatchesInputs(args []ValueDescr) bool {
	if k.varArgs {
		for i, arg := range args {
			if !k.inTypes[utils.MinInt(i, len(k.inTypes)-1)].Matches(arg) {
				return false
			}
		}
		return true
	}

	if len(args) != len(k.inTypes) {
		return false
	}

	for i, arg := range args {
		if !k.inTypes[i].Matches(arg) {
			return false
		}
	}
	return true
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

type KernelInitArgs struct {
	Kernel  Kernel
	Inputs  []ValueDescr
	Options FunctionOptions
}

type KernelState interface{}

type KernelInit func(*KernelCtx, KernelInitArgs) (KernelState, error)

type ArrayKernelExec func(*KernelCtx, *ExecBatch, Datum) error

type Kernel interface {
	GetInit() KernelInit
	GetSignature() *KernelSig
	GetNullHandling() NullHandling
	GetMemAlloc() MemAlloc
	CanWriteSlices() bool
	Execute(*KernelCtx, *ExecBatch, Datum) error
}

type kernelCtxKey struct{}

func GetKernelCtx(ctx context.Context) *KernelCtx {
	if v, ok := ctx.Value(kernelCtxKey{}).(*KernelCtx); ok {
		return v
	}
	return nil
}

func SetKernelCtx(ctx context.Context, kctx *KernelCtx) context.Context {
	return context.WithValue(ctx, kernelCtxKey{}, kctx)
}
