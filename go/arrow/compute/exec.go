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

// +build cgo

package compute

// #cgo !windows pkg-config: arrow
// #cgo windows LDFLAGS: -larrow -static -lole32
// #include "abi.h"
import "C"

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"unsafe"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/cdata"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/apache/arrow/go/arrow/scalar"
	"golang.org/x/xerrors"
)

type execCtxKey struct{}

func ExecContext(ctx context.Context) context.Context {
	ec := C.arrow_compute_default_context()
	return context.WithValue(ctx, execCtxKey{}, ec)
}

func ReleaseContext(ctx context.Context) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	C.arrow_compute_release_context(ec)
}

func convCArr(arr *C.struct_ArrowArray) *cdata.CArrowArray {
	return (*cdata.CArrowArray)(unsafe.Pointer(arr))
}

func convCSchema(schema *C.struct_ArrowSchema) *cdata.CArrowSchema {
	return (*cdata.CArrowSchema)(unsafe.Pointer(schema))
}

func ExecuteScalarExprWithSchema(ctx context.Context, mem memory.Allocator, schema *arrow.Schema, rb array.Record, expr Expression) (out Datum, err error) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	input := C.get_io()
	defer C.release_io(&input)

	cdata.ExportArrowRecordBatch(rb, convCArr(input.data), convCSchema(input.schema))
	defer cdata.CArrowArrayRelease(convCArr(input.data))
	defer cdata.CArrowSchemaRelease(convCSchema(input.schema))

	var fullSchema C.struct_ArrowSchema
	cdata.ExportArrowSchema(schema, convCSchema(&fullSchema))
	defer cdata.CArrowSchemaRelease(convCSchema(&fullSchema))

	output := C.get_io()
	defer C.release_io(&output)

	if ec := C.arrow_compute_execute_scalar_expr_schema(ec, &fullSchema, &input, C.BoundExpression(expr.bound()), &output); ec != 0 {
		return nil, xerrors.Errorf("got errorcode: %d", ec)
	}

	defer cdata.CArrowArrayRelease(convCArr(output.data))
	defer cdata.CArrowSchemaRelease(convCSchema(output.schema))

	f, arr, err := cdata.ImportCArray((*cdata.CArrowArray)(unsafe.Pointer(output.data)), (*cdata.CArrowSchema)(unsafe.Pointer(output.schema)))
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	switch {
	case arr.Len() == 1:
		sc, err := scalar.GetScalar(arr, 0)
		if err != nil {
			return nil, err
		}
		return NewDatum(sc), nil
	case f.Type.ID() == arrow.STRUCT:
		st := arr.(*array.Struct)
		cols := make([]array.Interface, st.NumField())
		for i := 0; i < st.NumField(); i++ {
			cols[i] = st.Field(i)
		}
		rec := array.NewRecord(arrow.NewSchema(st.DataType().(*arrow.StructType).Fields(), &f.Metadata), cols, int64(cols[0].Len()))
		defer rec.Release()
		return NewDatum(rec), nil
	default:
		return NewDatum(arr), nil
	}
}

func ExecuteScalarExpr(ctx context.Context, mem memory.Allocator, rb array.Record, expr Expression) (out Datum, err error) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	input := C.get_io()
	defer C.release_io(&input)

	cdata.ExportArrowRecordBatch(rb, convCArr(input.data), convCSchema(input.schema))
	defer cdata.CArrowArrayRelease(convCArr(input.data))
	defer cdata.CArrowSchemaRelease(convCSchema(input.schema))

	output := C.get_io()
	defer C.release_io(&output)

	buf := SerializeExpr(expr, mem)
	defer buf.Release()
	if ec := C.arrow_compute_execute_scalar_expr(ec, &input, (*C.uint8_t)(unsafe.Pointer(&buf.Bytes()[0])), C.int(buf.Len()), &output); ec != 0 {
		return nil, xerrors.Errorf("got errorcode: %d", ec)
	}
	defer cdata.CArrowArrayRelease(convCArr(output.data))
	defer cdata.CArrowSchemaRelease(convCSchema(output.schema))

	f, arr, err := cdata.ImportCArray((*cdata.CArrowArray)(unsafe.Pointer(output.data)), (*cdata.CArrowSchema)(unsafe.Pointer(output.schema)))
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	if f.Type.ID() == arrow.STRUCT {
		st := arr.(*array.Struct)
		cols := make([]array.Interface, st.NumField())
		for i := 0; i < st.NumField(); i++ {
			cols[i] = st.Field(i)
		}
		rec := array.NewRecord(arrow.NewSchema(st.DataType().(*arrow.StructType).Fields(), &f.Metadata), cols, int64(cols[0].Len()))
		defer rec.Release()
		return NewDatum(rec), nil
	}
	return NewDatum(arr), nil
}

func isFuncScalar(funcName string) bool {
	cfuncName := C.CString(funcName)
	defer C.free(unsafe.Pointer(cfuncName))

	return C.arrow_compute_function_scalar(cfuncName) == C._Bool(true)
}

type boundRef C.BoundExpression

func (b boundRef) isScalar() bool {
	return C.arrow_compute_bound_is_scalar(C.BoundExpression(b)) == C._Bool(true)
}

func (b boundRef) isSatisfiable() bool {
	return C.arrow_compute_bound_is_satisfiable(C.BoundExpression(b)) == C._Bool(true)
}

func (b boundRef) release() {
	C.arrow_compute_bound_expr_release(C.BoundExpression(b))
}

func getBoundType(b boundRef) (arrow.DataType, error) {
	var cschema C.struct_ArrowSchema
	C.arrow_compute_bound_expr_type(C.BoundExpression(b), &cschema)
	field, err := cdata.ImportCArrowField(convCSchema(&cschema))
	if err != nil {
		return nil, err
	}
	return field.Type, nil
}

func copyForBind(expr Expression, boundExpr boundRef) (out Expression) {
	switch e := expr.(type) {
	case *Literal:
		out = &Literal{e.Literal, boundExpr}
	case *Parameter:
		dt, err := getBoundType(boundExpr)
		if err != nil {
			panic(err)
		}
		out = &Parameter{ref: e.ref, index: e.index, descr: ValueDescr{ShapeArray, dt}}
	case *Call:
		dt, err := getBoundType(boundExpr)
		if err != nil {
			panic(err)
		}
		var call Call = *e
		call.descr = ValueDescr{Type: dt}
		if boundExpr.isScalar() {
			call.descr.Shape = ShapeScalar
		} else {
			call.descr.Shape = ShapeArray
		}
		call.b = boundExpr
		out = &call
	}

	runtime.SetFinalizer(out, func(o Expression) {
		var b boundRef
		switch o := o.(type) {
		case *Literal:
			b = o.b
		case *Parameter:
			b = o.b
		case *Call:
			b = o.b
		}
		b.release()
	})
	return
}

func getBoundArg(b boundRef, i int, expr Expression) Expression {
	bound := C.arrow_compute_get_bound_arg(C.BoundExpression(b), C.size_t(i))
	return copyForBind(expr, boundRef(bound))
}

func BindExpression(ctx context.Context, mem memory.Allocator, expr Expression, schema *arrow.Schema) Expression {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	buf := SerializeExpr(expr, mem)
	defer buf.Release()

	var cschema C.struct_ArrowSchema
	cdata.ExportArrowSchema(schema, convCSchema(&cschema))
	defer cdata.CArrowSchemaRelease(convCSchema(&cschema))

	boundExpr := boundRef(C.arrow_compute_bind_expr(ec, &cschema, (*C.uint8_t)(unsafe.Pointer(&buf.Bytes()[0])), C.int(buf.Len())))

	return copyForBind(expr, boundExpr)
}

func SimplifyWithGuarantee(expr, guaranteedTruePred Expression) (Expression, error) {
	if !expr.IsBound() {
		return nil, errors.New("must provide bound expr")
	}

	var bound boundRef
	switch e := expr.(type) {
	case *Literal:
		bound = e.b
	case *Call:
		bound = e.b
	case *Parameter:
		bound = e.b
	case *unknownBoundExpr:
		bound = e.b
	}

	var out C.BoundExpression
	buf := SerializeExpr(guaranteedTruePred, memory.DefaultAllocator)
	defer buf.Release()

	if C.arrow_compute_bound_expr_simplify_guarantee(C.BoundExpression(bound), (*C.uint8_t)(unsafe.Pointer(&buf.Bytes()[0])), C.int(buf.Len()), &out) != 0 {
		return nil, errors.New("bad expr simplify")
	}

	return &unknownBoundExpr{b: boundRef(out)}, nil
}

func CallFunction(ctx context.Context, mem memory.Allocator, funcName string, args []Datum, options *FunctionOptions) (Datum, error) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)

	argsIO := C.get_io()
	defer C.release_io(&argsIO)

	optionsIO := C.get_io()
	defer C.release_io(&optionsIO)

	output := C.get_io()
	defer C.release_io(&output)

	const datumtypekey = "arrow::datum::type"

	fields := make([]arrow.Field, len(args))
	cols := make([]array.Interface, len(args))
	for i, a := range args {
		var arr array.Interface
		var val string
		switch a.Kind() {
		case KindRecordBatch:
			val = "record"
			arr = array.RecordToStructArray(a.(*RecordDatum).Value)
		case KindArray:
			val = "array"
			arr = a.(*ArrayDatum).MakeArray()
		}

		defer arr.Release()
		cols[i] = arr
		fields[i] = arrow.Field{Name: strconv.Itoa(i), Type: arr.DataType(), Nullable: true, Metadata: arrow.NewMetadata([]string{datumtypekey}, []string{val})}
	}

	argRec := array.NewRecord(arrow.NewSchema(fields, nil), cols, -1)
	defer argRec.Release()

	cdata.ExportArrowRecordBatch(argRec, convCArr(argsIO.data), convCSchema(argsIO.schema))
	defer cdata.CArrowArrayRelease(convCArr(argsIO.data))
	defer cdata.CArrowSchemaRelease(convCSchema(argsIO.schema))

	optscalar, err := options.ToStructScalar(mem)
	if err != nil {
		return nil, err
	}

	optarr, err := scalar.MakeArrayFromScalar(optscalar, 1, mem)
	if err != nil {
		return nil, err
	}
	defer optarr.Release()

	cdata.ExportArrowArray(optarr, convCArr(optionsIO.data), convCSchema(optionsIO.schema))

	cfuncName := C.CString(funcName)
	defer C.free(unsafe.Pointer(cfuncName))

	if ec := C.call_function(ec, cfuncName, &argsIO, &optionsIO, &output); ec != 0 {
		return nil, fmt.Errorf("got error code: %d", ec)
	}

	defer cdata.CArrowArrayRelease(convCArr(output.data))
	defer cdata.CArrowSchemaRelease(convCSchema(output.schema))

	f, arr, err := cdata.ImportCArray((*cdata.CArrowArray)(unsafe.Pointer(output.data)), (*cdata.CArrowSchema)(unsafe.Pointer(output.schema)))
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	switch f.Name {
	case "scalar":
		sc, err := scalar.GetScalar(arr, 0)
		if err != nil {
			return nil, err
		}
		return NewDatum(sc), nil
	case "array":
		return NewDatum(arr), nil
	default:
		st := arr.(*array.Struct)
		cols := make([]array.Interface, st.NumField())
		for i := 0; i < st.NumField(); i++ {
			cols[i] = st.Field(i)
		}
		rec := array.NewRecord(arrow.NewSchema(st.DataType().(*arrow.StructType).Fields(), &f.Metadata), cols, int64(cols[0].Len()))
		defer rec.Release()
		return NewDatum(rec), nil
	}
}

func Filter(ctx context.Context, mem memory.Allocator, in, mask Datum, opts FilterOptions) (Datum, error) {
	return CallFunction(ctx, mem, "filter", []Datum{in, mask}, NewFunctionOption(opts))
}
