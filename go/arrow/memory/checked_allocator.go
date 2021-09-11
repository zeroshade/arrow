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

package memory

import (
	"runtime"
	"sync"
	"unsafe"
)

// typically the allocations are happening in memory.Buffer, not by consumers calling
// allocate/reallocate directly. As a result, we want to skip the caller frames
// of the inner workings of Buffer in order to find the caller that actually triggered
// the allocation via a call to Resize/Reserve/etc.
const (
	allocFrames   = 4
	reallocFrames = 3
)

type dalloc struct {
	pc   uintptr
	line int
	sz   int
}

type CheckedAllocator struct {
	mem Allocator
	sz  int

	allocs sync.Map
}

func NewCheckedAllocator(mem Allocator) *CheckedAllocator {
	return &CheckedAllocator{mem: mem}
}

func (a *CheckedAllocator) Allocate(size int) []byte {
	a.sz += size
	out := a.mem.Allocate(size)
	ptr := uintptr(unsafe.Pointer(&out[0]))
	pc, _, l, ok := runtime.Caller(allocFrames)
	if ok {
		a.allocs.Store(ptr, &dalloc{pc: pc, line: l, sz: size})
	}
	return out
}

func (a *CheckedAllocator) Reallocate(size int, b []byte) []byte {
	a.sz += size - len(b)

	oldptr := uintptr(unsafe.Pointer(&b[0]))
	out := a.mem.Reallocate(size, b)
	newptr := uintptr(unsafe.Pointer(&out[0]))

	a.allocs.Delete(oldptr)
	pc, _, l, ok := runtime.Caller(reallocFrames)
	if ok {
		a.allocs.Store(newptr, &dalloc{pc: pc, line: l, sz: size})
	}
	return out
}

func (a *CheckedAllocator) Free(b []byte) {
	a.sz -= len(b)
	defer a.mem.Free(b)

	if len(b) == 0 {
		return
	}

	ptr := uintptr(unsafe.Pointer(&b[0]))
	a.allocs.Delete(ptr)
}

type TestingT interface {
	Logf(format string, args ...interface{})
	Errorf(format string, args ...interface{})
	Helper()
}

func (a *CheckedAllocator) AssertSize(t TestingT, sz int) {
	a.allocs.Range(func(_, value interface{}) bool {
		info := value.(*dalloc)
		f := runtime.FuncForPC(info.pc)
		t.Errorf("LEAK of %d bytes FROM %s line %d\n", info.sz, f.Name(), info.line)
		return true
	})

	if a.sz != sz {
		t.Helper()
		t.Errorf("invalid memory size exp=%d, got=%d", sz, a.sz)
	}
}

type CheckedAllocatorScope struct {
	alloc *CheckedAllocator
	sz    int
}

func NewCheckedAllocatorScope(alloc *CheckedAllocator) *CheckedAllocatorScope {
	return &CheckedAllocatorScope{alloc: alloc, sz: alloc.sz}
}

func (c *CheckedAllocatorScope) CheckSize(t TestingT) {
	if c.sz != c.alloc.sz {
		t.Helper()
		t.Errorf("invalid memory size exp=%d, got=%d", c.sz, c.alloc.sz)
	}
}

var (
	_ Allocator = (*CheckedAllocator)(nil)
)
