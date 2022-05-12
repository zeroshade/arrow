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
	"fmt"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
)

func getDescriptors(exprs []compute.Expression) []compute.ValueDescr {
	out := make([]compute.ValueDescr, len(exprs))
	for i, ex := range exprs {
		debug.Assert(ex.IsBound(), "should be bound before calling getDescriptors")
		out[i] = ex.Descr()
	}
	return out
}

func bindNonRecurse(ectx *compute.ExecCtx, call *compute.Call, insertImplicitCasts bool) (compute.Expression, error) {
	descrs := getDescriptors(call.Args())
	var err error
	call.BoundFunc, err = ectx.Registry.GetFunction(call.FunctionName())
	if err != nil {
		return nil, err
	}

	if !insertImplicitCasts {
		call.Kernel, err = call.BoundFunc.DispatchExact(descrs)
		if err != nil {
			return nil, err
		}
	} else {
		call.Kernel, err = call.BoundFunc.DispatchBest(descrs)
		if err != nil {
			return nil, err
		}

		ctx := compute.SetExecCtx(context.Background(), ectx)

		args := call.Args()
		for i, desc := range descrs {
			if args[i].Descr().Equals(desc) {
				continue
			}

			if args[i].Descr().Shape != desc.Shape {
				return nil, fmt.Errorf("%w: automatic broadcasting of scalar arguments to arrays in %s", compute.ErrNotImplemented, call)
			}

			if lit, ok := args[i].(*compute.Literal); ok {
				newLit, err := CastTo(ctx, lit.Literal, desc.Type, *compute.DefaultCastOptions(true))
				if err != nil {
					return nil, err
				}
				defer newLit.Release()
				defer lit.Release()
				args[i] = compute.NewLiteral(newLit)
				continue
			}

			// construct an implicit cast expression and replace
			implicitCast := compute.Cast(args[i], desc.Type).(*compute.Call)
			defer implicitCast.Release()

			args[i], err = bindNonRecurse(ectx, implicitCast, false)
			if err != nil {
				return nil, err
			}
		}
	}

	kctx := compute.KernelCtx{Ctx: ectx}
	initFn := call.Kernel.GetInit()
	if initFn != nil {
		opts := call.Options()
		if opts == nil {
			opts = call.BoundFunc.DefaultOptions()
		}
		call.KernelState, err = initFn(&kctx, compute.KernelInitArgs{
			Kernel: call.Kernel, Inputs: descrs, Options: opts})
		if err != nil {
			return nil, err
		}
		kctx.State = call.KernelState
	}

	call.BoundDescr, err = call.Kernel.GetSignature().OutputType().Resolve(&kctx, descrs)
	return call, err
}

func bindImpl(ectx *compute.ExecCtx, expr compute.Expression, sc *arrow.Schema, shape compute.ValueShape) (compute.Expression, error) {
	switch ex := expr.(type) {
	case *compute.Literal:
		return expr, nil
	case *compute.Parameter:
		ref := ex.FieldRef()
		path, err := ref.FindOne(sc)
		if err != nil {
			return nil, err
		}

		field, err := path.Get(sc)
		if err != nil {
			return nil, err
		}

		return compute.NewBoundRef(ref, path, compute.ValueDescr{Shape: shape, Type: field.Type}), nil
	case *compute.Call:
		args := ex.Args()
		boundArgs := make([]compute.Expression, len(args))
		var err error
		for i, arg := range args {
			if boundArgs[i], err = bindImpl(ectx, arg, sc, shape); err != nil {
				return nil, err
			}
		}
		return bindNonRecurse(ectx, compute.NewCall(ex.FunctionName(), boundArgs, ex.Options()), true)
	}
	return nil, fmt.Errorf("%w bad expression type", compute.ErrInvalid)
}

func BindSchema(ctx context.Context, expr compute.Expression, sc *arrow.Schema) (compute.Expression, error) {
	ectx := compute.GetExecCtx(ctx)
	if ectx == nil {
		ectx = DefaultExecCtx()
	}
	return bindImpl(ectx, expr, sc, compute.ShapeArray)
}
