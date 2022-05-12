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

	"github.com/apache/arrow/go/v9/arrow/compute"
)

type ExecutableFunc interface {
	compute.Function
	Execute(ctx context.Context, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error)
}

type baseFunc struct {
	name    string
	kind    compute.FunctionKind
	arity   compute.Arity
	options compute.FunctionOptions
	doc     compute.FunctionDoc
}

func (b *baseFunc) Name() string                            { return b.name }
func (b *baseFunc) Kind() compute.FunctionKind              { return b.kind }
func (b *baseFunc) Arity() compute.Arity                    { return b.arity }
func (b *baseFunc) Doc() compute.FunctionDoc                { return b.doc }
func (b *baseFunc) DefaultOptions() compute.FunctionOptions { return b.options }

func validArity(f *baseFunc, numArgs int, label string) error {
	switch {
	case f.arity.VarArgs && numArgs < f.arity.NumArgs:
		return fmt.Errorf("varargs function '%s' needs at least %d arguments, but %s only %d",
			f.name, f.arity.NumArgs, label, numArgs)
	case !f.arity.VarArgs && numArgs != f.arity.NumArgs:
		return fmt.Errorf("function '%s' accepts %d args but %s %d",
			f.name, f.arity.NumArgs, label, numArgs)
	default:
		return nil
	}
}

func (b *baseFunc) CheckArityTypes(in []compute.InputType) error {
	return validArity(b, len(in), "kernel accepts")
}
func (b *baseFunc) CheckArityDescr(in []compute.ValueDescr) error {
	return validArity(b, len(in), "attempted to look up kernel(s) with")
}

func (b *baseFunc) DispatchExact(vals []compute.ValueDescr) (compute.Kernel, error) {
	if b.kind == compute.FuncMetaKind {
		return nil, errors.New("not implemented dispatch for metafunction kernel")
	}

	return nil, nil
}

func (b *baseFunc) DispatchBest(vals []compute.ValueDescr) (compute.Kernel, error) {
	return b.DispatchExact(vals)
}

type MetaFunction struct {
	baseFunc

	impl func(context.Context, []compute.Datum, compute.FunctionOptions) (compute.Datum, error)
}

func NewMetaFunction(name string, arity compute.Arity, doc compute.FunctionDoc, impl func(context.Context, []compute.Datum, compute.FunctionOptions) (compute.Datum, error)) MetaFunction {
	return MetaFunction{
		baseFunc: baseFunc{name: name, arity: arity, doc: doc, kind: compute.FuncMetaKind},
		impl:     impl,
	}
}

func (mf *MetaFunction) Execute(ctx context.Context, args []compute.Datum, opts compute.FunctionOptions) (compute.Datum, error) {
	if err := validArity(&mf.baseFunc, len(args), "attempted to execute with"); err != nil {
		return nil, err
	}

	if opts == nil {
		opts = mf.options
	}

	return mf.impl(ctx, args, opts)
}

type ScalarFunction struct {
	baseFunc

	kernels []ScalarKernel
}

func NewScalarFunction(name string, arity compute.Arity, defaultOpts compute.FunctionOptions) ScalarFunction {
	return ScalarFunction{
		baseFunc: baseFunc{name: name, arity: arity, options: defaultOpts, kind: compute.FuncScalarKind},
		kernels:  make([]ScalarKernel, 0),
	}
}

func (sf *ScalarFunction) Kernels() []ScalarKernel { return sf.kernels }

func (sf *ScalarFunction) AddNewKernel(in []compute.InputType, out compute.OutputType, exec ArrayKernelExec, init compute.KernelInit) error {
	if err := sf.CheckArityTypes(in); err != nil {
		return err
	}

	if sf.arity.VarArgs && len(in) != 1 {
		return errors.New("scalar varargs signatures must have exactly one input type")
	}
	sf.kernels = append(sf.kernels, NewScalarKernel(in, out, exec, init))
	return nil
}

func (sf *ScalarFunction) AddKernel(kernel ScalarKernel) error {
	if err := sf.CheckArityTypes(kernel.Signature.InputTypes()); err != nil {
		return err
	}

	if sf.arity.VarArgs && !kernel.Signature.VarArgs() {
		return errors.New("function accepts varargs but kernel signature does not")
	}
	sf.kernels = append(sf.kernels, kernel)
	return nil
}
