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
	"github.com/apache/arrow/go/v9/arrow/compute"
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

type kernel struct {
	Init           compute.KernelInit
	Parallelizable bool
	SimdLevel      MinSIMDLevel
	Signature      *compute.KernelSig
}

func (k *kernel) GetSignature() *compute.KernelSig { return k.Signature }
func (k *kernel) GetInit() compute.KernelInit      { return k.Init }

func newKernel(inTypes []compute.InputType, out compute.OutputType, init compute.KernelInit) kernel {
	return kernel{
		Init:           init,
		Signature:      compute.NewKernelSig(inTypes, out, false),
		Parallelizable: true,
	}
}

type ArrayKernelExec func(*compute.KernelCtx, *compute.ExecBatch, compute.Datum) error

type ScalarKernel struct {
	kernel

	Exec               ArrayKernelExec
	CanWriteIntoSlices bool
	NullHandling       compute.NullHandling
	MemAlloc           compute.MemAlloc
}

func (s *ScalarKernel) GetNullHandling() compute.NullHandling {
	return s.NullHandling
}

func (s *ScalarKernel) GetMemAlloc() compute.MemAlloc { return s.MemAlloc }

func (s *ScalarKernel) CanWriteSlices() bool { return s.CanWriteIntoSlices }

func (s *ScalarKernel) Execute(ctx *compute.KernelCtx, batch *compute.ExecBatch, result compute.Datum) error {
	return s.Exec(ctx, batch, result)
}

func NewScalarKernel(in []compute.InputType, out compute.OutputType, exec ArrayKernelExec, init compute.KernelInit) ScalarKernel {
	return ScalarKernel{
		kernel:             newKernel(in, out, init),
		Exec:               exec,
		CanWriteIntoSlices: true,
		NullHandling:       compute.NullIntersection,
		MemAlloc:           compute.MemPrealloc,
	}
}

func NewScalarKernelWithSig(sig *compute.KernelSig, exec ArrayKernelExec, init compute.KernelInit) ScalarKernel {
	return ScalarKernel{
		kernel:             kernel{Signature: sig, Init: init, Parallelizable: true},
		Exec:               exec,
		CanWriteIntoSlices: true,
		NullHandling:       compute.NullIntersection,
		MemAlloc:           compute.MemPrealloc,
	}
}
