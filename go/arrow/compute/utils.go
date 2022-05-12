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
	"io"
	"math"
	"sync"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/memory"
	"golang.org/x/xerrors"
)

type RegistryFactory func() FunctionRegistry

var (
	factory      RegistryFactory
	initRegistry sync.Once
	registry     FunctionRegistry
)

func SetRegistryFactory(fn RegistryFactory) {
	factory = fn
}

func GetRegistry() FunctionRegistry {
	initRegistry.Do(func() {
		if factory == nil {
			return
		}
		registry = factory()
	})
	return registry
}

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

type Arity struct {
	NumArgs int
	VarArgs bool
}

func Nullary() Arity            { return Arity{0, false} }
func Unary() Arity              { return Arity{1, false} }
func Binary() Arity             { return Arity{2, false} }
func Ternary() Arity            { return Arity{3, false} }
func VarArgs(minargs int) Arity { return Arity{minargs, true} }

type FunctionKind int8

const (
	FuncScalarKind FunctionKind = iota
	FuncVectorKind
	FuncScalarAggKind
	FuncHashAggKind
	FuncMetaKind
)

type FunctionDoc struct {
	Summary         string
	Desc            string
	ArgNames        []string
	OptionsType     string
	OptionsRequired bool
}

type Function interface {
	Name() string
	Kind() FunctionKind
	Arity() Arity
	Doc() FunctionDoc
	DispatchExact([]ValueDescr) (Kernel, error)
	DispatchBest([]ValueDescr) (Kernel, error)
	DefaultOptions() FunctionOptions
}

type FunctionRegistry interface {
	AddFunction(fn Function, allowOverwrite bool) error
	AddAlias(target, source string) error
	GetFunction(name string) (Function, error)
	GetFunctionNames() []string
}

type execCtxKey struct{}

func SetExecCtx(ctx context.Context, ectx *ExecCtx) context.Context {
	return context.WithValue(ctx, execCtxKey{}, ectx)
}

func GetExecCtx(ctx context.Context) *ExecCtx {
	if ec, ok := ctx.Value(execCtxKey{}).(*ExecCtx); ok {
		return ec
	}
	return nil
}

type ExecCtx struct {
	Mem                memory.Allocator
	ChunkSize          int64
	PreallocContiguous bool
	Registry           FunctionRegistry
}

type bufferWriteSeeker struct {
	buf *memory.Buffer
	pos int
	mem memory.Allocator
}

func (b *bufferWriteSeeker) Reserve(nbytes int) {
	if b.buf == nil {
		b.buf = memory.NewResizableBuffer(b.mem)
	}
	newCap := int(math.Max(float64(b.buf.Cap()), 256))
	for newCap < b.pos+nbytes {
		newCap = bitutil.NextPowerOf2(newCap)
	}
	b.buf.Reserve(newCap)
}

func (b *bufferWriteSeeker) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	if b.buf == nil {
		b.Reserve(len(p))
	} else if b.pos+len(p) >= b.buf.Cap() {
		b.Reserve(len(p))
	}

	return b.UnsafeWrite(p)
}

func (b *bufferWriteSeeker) UnsafeWrite(p []byte) (n int, err error) {
	n = copy(b.buf.Buf()[b.pos:], p)
	b.pos += len(p)
	if b.pos > b.buf.Len() {
		b.buf.ResizeNoShrink(b.pos)
	}
	return
}

func (b *bufferWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	newpos, offs := 0, int(offset)
	switch whence {
	case io.SeekStart:
		newpos = offs
	case io.SeekCurrent:
		newpos = b.pos + offs
	case io.SeekEnd:
		newpos = b.buf.Len() + offs
	}
	if newpos < 0 {
		return 0, xerrors.New("negative result pos")
	}
	b.pos = newpos
	return int64(newpos), nil
}
