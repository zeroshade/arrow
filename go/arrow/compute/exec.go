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
	"runtime"
	"unsafe"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/cdata"
	"github.com/apache/arrow/go/arrow/memory"
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

// type BoundExpression struct {
// 	b  C.BoundExpression
// 	dt arrow.DataType

// 	origExpr Expression
// 	hash     uint64
// }

// func (b BoundExpression) IsBound() bool {
// 	return b.b != 0
// }

// func (b BoundExpression) IsScalarExpr() bool {
// 	if _, ok := b.origExpr.(*Call); ok {
// 		return C.arrow_compute_bound_is_scalar(b.b) == C._Bool(true)
// 	}
// 	return b.origExpr.IsScalarExpr()
// }

// func (b BoundExpression) IsNullLiteral() bool {
// 	return b.origExpr.IsNullLiteral()
// }

// func (b *BoundExpression) IsSatisfiable() bool {
// 	if lit, ok := b.origExpr.(*Literal); ok {
// 		return lit.IsSatisfiable()
// 	}

// 	dt := b.Type()
// 	return dt == nil || dt.ID() != arrow.NULL
// }

// func (b *BoundExpression) FieldRef() *FieldRef {
// 	return b.origExpr.FieldRef()
// }

// func (b *BoundExpression) Descr() ValueDescr {
// 	switch e := b.origExpr.(type) {
// 	case *Literal:
// 		return e.Descr()
// 	case *Parameter:
// 		return ValueDescr{Shape: ShapeArray, Type: b.Type()}
// 	case *Call:
// 		if b.IsScalarExpr() {
// 			return ValueDescr{Shape: ShapeScalar, Type: b.Type()}
// 		}
// 		return ValueDescr{Shape: ShapeArray, Type: b.Type()}
// 	}
// 	return ValueDescr{}
// }

// func (b BoundExpression) Equals(rhs Expression) bool {
// 	return b.origExpr.Equals(rhs)
// }

// func (b BoundExpression) Hash() uint64 {
// 	return b.hash
// }

// func (b BoundExpression) Release() {
// 	C.arrow_compute_bound_expr_release(b.b)
// }

// func (b *BoundExpression) Type() arrow.DataType {
// 	dt, err := b.DataType()
// 	if err != nil {
// 		panic(err)
// 	}
// 	return dt
// }

// func (b *BoundExpression) DataType() (arrow.DataType, error) {
// 	if b.dt == nil {
// 		var cschema C.struct_ArrowSchema
// 		C.arrow_compute_bound_expr_type(b.b, &cschema)
// 		field, err := cdata.ImportCArrowField(convCSchema(&cschema))
// 		if err != nil {
// 			return nil, err
// 		}
// 		b.dt = field.Type
// 	}
// 	return b.dt, nil
// }

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
