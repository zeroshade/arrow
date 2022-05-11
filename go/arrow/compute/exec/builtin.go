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

package exec

import (
	"context"
	"math"
	"sync"

	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/kernels"
	"github.com/apache/arrow/go/v9/arrow/memory"
)

var (
	initRegistry sync.Once
	registry     *functions.FunctionRegistry

	defaultExecCtx     *functions.ExecCtx
	initDefaultExecCtx sync.Once
)

func GetRegistry() *functions.FunctionRegistry {
	initRegistry.Do(func() {
		registry = &functions.FunctionRegistry{}
		kernels.RegisterScalarCasts(registry)
	})
	return registry
}

func DefaultExecCtx() *functions.ExecCtx {
	initDefaultExecCtx.Do(func() {
		defaultExecCtx = &functions.ExecCtx{
			Mem:                memory.DefaultAllocator,
			ChunkSize:          math.MaxInt64,
			PreallocContiguous: true,
			Registry:           GetRegistry(),
		}
	})
	return defaultExecCtx
}

func CallFunction(ctx context.Context, funcname string, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
	ectx := functions.GetExecCtx(ctx)
	if ectx == nil {
		return CallFunction(functions.SetExecCtx(ctx, DefaultExecCtx()), funcname, args, opts)
	}

	fn, err := ectx.Registry.GetFunction(funcname)
	if err != nil {
		return nil, err
	}
	return functions.ExecuteFunction(ctx, fn, args, opts)
}
