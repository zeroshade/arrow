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

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/kernels"
	"github.com/apache/arrow/go/v9/arrow/memory"
)

var (
	defaultExecCtx     *compute.ExecCtx
	initDefaultExecCtx sync.Once
)

func init() {
	compute.SetRegistryFactory(func() compute.FunctionRegistry {
		registry := &functions.FunctionRegistry{}
		kernels.RegisterScalarCasts(registry)
		return registry
	})
}

func DefaultExecCtx() *compute.ExecCtx {
	initDefaultExecCtx.Do(func() {
		defaultExecCtx = &compute.ExecCtx{
			Mem:                memory.DefaultAllocator,
			ChunkSize:          math.MaxInt64,
			PreallocContiguous: true,
			Registry:           compute.GetRegistry(),
		}
	})
	return defaultExecCtx
}

func CallFunction(ctx context.Context, funcname string, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
	ectx := compute.GetExecCtx(ctx)
	if ectx == nil {
		return CallFunction(compute.SetExecCtx(ctx, DefaultExecCtx()), funcname, args, opts)
	}

	fn, err := ectx.Registry.GetFunction(funcname)
	if err != nil {
		return nil, err
	}
	return functions.ExecuteFunction(ctx, fn, args, opts)
}

func Cast(ctx context.Context, value compute.Datum, options *compute.CastOptions) (compute.Datum, error) {
	return CallFunction(ctx, "cast", []compute.Datum{value}, options)
}

func CastTo(ctx context.Context, value compute.Datum, toType arrow.DataType, options compute.CastOptions) (compute.Datum, error) {
	options.ToType = toType
	return Cast(ctx, value, &options)
}

func CastArray(ctx context.Context, value arrow.Array, to arrow.DataType, options *compute.CastOptions) (arrow.Array, error) {
	datum := compute.NewDatum(value)
	defer datum.Release()

	out, err := CastTo(ctx, datum, to, *options)
	if err != nil {
		return nil, err
	}
	defer out.Release()

	return out.(*compute.ArrayDatum).MakeArray(), nil
}
