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

// +build ccalloc ccexec

package compute

// #cgo !windows pkg-config: arrow
// #cgo CXXFLAGS: -std=c++14
// #cgo windows LDFLAGS: -larrow
// #include "exec.h"
// #include <stdlib.h>
import "C"
import (
	"context"
	"errors"
	"reflect"
	"unsafe"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/cdata"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/apache/arrow/go/arrow/scalar"
)

func isFuncScalar(funcName string) bool {
	cfuncName := C.CString(funcName)
	defer C.free(unsafe.Pointer(cfuncName))

	return C.arrow_compute_function_is_scalar(cfuncName) == C._Bool(true)
}

type execCtxKey struct{}

func WithExecCtx(ctx context.Context) context.Context {
	ec := C.arrow_compute_default_context()
	return context.WithValue(ctx, execCtxKey{}, ec)
}

func ReleaseExecContext(ctx context.Context) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	C.arrow_compute_release_context(ec)
}

func GetExecChunkSize(ctx context.Context) int64 {
	ec, ok := ctx.Value(execCtxKey{}).(C.ExecContext)
	if !ok {
		return int64(C.kDefaultExecChunk)
	}
	return int64(C.arrow_compute_get_exec_chunksize(ec))
}

func SetExecChunkSize(ctx context.Context, size int64) {
	ec, ok := ctx.Value(execCtxKey{}).(C.ExecContext)
	if !ok {
		panic("arrow/compute: cannot set chunk size on null exec context")
	}
	C.arrow_compute_set_exec_chunksize(ec, C.int64_t(size))
}

type boundRef C.BoundExpression

func (b boundRef) release() {
	C.arrow_compute_bound_expr_release(C.BoundExpression(b))
}

func (b boundRef) gettype() (arrow.DataType, error) {
	cschema := (*C.struct_ArrowSchema)(C.malloc(C.sizeof_struct_ArrowSchema))
	defer C.free(unsafe.Pointer(cschema))
	C.arrow_compute_bound_expr_type(C.BoundExpression(b), cschema)
	field, err := cdata.ImportCArrowField(cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cschema))))
	if err != nil {
		return nil, err
	}
	return field.Type, nil
}

func bindExprSchema(ctx context.Context, mem memory.Allocator, expr Expression, schema *arrow.Schema) (boundRef, ValueDescr, int, error) {
	ec, _ := ctx.Value(execCtxKey{}).(C.ExecContext)
	// if there's no context, it'll be 0 and we'll just use null in the c++
	// and get the default context there.

	buf, err := SerializeExpr(expr, mem)
	if err != nil {
		return 0, ValueDescr{}, 0, err
	}
	defer buf.Release()

	cschema := (*C.struct_ArrowSchema)(C.malloc(C.sizeof_struct_ArrowSchema))
	cdata.ExportArrowSchema(schema, cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cschema))))
	defer C.free(unsafe.Pointer(cschema))

	postbind := C.arrow_compute_bind_expr(ec, cschema, (*C.uint8_t)(unsafe.Pointer(&buf.Bytes()[0])), C.int(buf.Len()))
	if postbind.bound == 0 {
		status := C.GoString(postbind.status)
		return 0, ValueDescr{}, 0, errors.New(status)
	}

	b := boundRef(postbind.bound)
	var descr ValueDescr
	if postbind._type != nil {
		defer C.free(unsafe.Pointer(postbind._type))
		field, err := cdata.ImportCArrowField(cdata.SchemaFromPtr(uintptr(unsafe.Pointer(postbind._type))))
		if err != nil {
			b.release()
			return 0, ValueDescr{}, 0, err
		}

		descr.Type = field.Type
		switch postbind.shape {
		case C.arrow_shape_any:
			descr.Shape = ShapeAny
		case C.arrow_shape_array:
			descr.Shape = ShapeArray
		case C.arrow_shape_scalar:
			descr.Shape = ShapeScalar
		}
	}

	return b, descr, int(postbind.index), nil
}

func ExecuteScalarExpression(ctx context.Context, expr Expression, mem memory.Allocator, schema *arrow.Schema, input Datum) (Datum, error) {
	if !expr.IsBound() || expr.boundExpr() == 0 {
		if lit, ok := expr.(*Literal); ok {
			// literals are always considered bound, we just need to do the binding
			// if the caller didn't.
			b, _, _, err := bindExprSchema(ctx, mem, expr, schema)
			if err != nil {
				return nil, err
			}
			lit.bound = b
		}
		return nil, errors.New("must pass a bound expression to execute")
	}

	ec, _ := ctx.Value(execCtxKey{}).(C.ExecContext)
	// if there's no context, it'll be 0 and we'll just use null in the c++
	// and get the default context there.

	in, err := datumToC(input, mem)
	if err != nil {
		return nil, err
	}
	defer freeArrowDatum(in)

	cschema := (*C.struct_ArrowSchema)(C.malloc(C.sizeof_struct_ArrowSchema))
	cdata.ExportArrowSchema(schema, cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cschema))))
	defer C.free(unsafe.Pointer(cschema))

	result := C.arrow_compute_exec_scalar_expr(C.BoundExpression(expr.boundExpr()), cschema, in, ec)
	defer freeArrowDatum(result)

	return datumFromC(result, mem)
}

func CallFunction(ctx context.Context, mem memory.Allocator, funcName string, args []Datum, options FunctionOptions) (Datum, error) {
	ec, _ := ctx.Value(execCtxKey{}).(C.ExecContext)
	// if there's no context, it'll be 0 and we'll just use null in the c++
	// and get the default context there.

	var (
		optbytes []byte
		coptname *C.char
	)
	var err error
	if options != nil {
		opts, err := SerializeOptions(options, mem)
		if err != nil {
			return nil, err
		}
		defer opts.Release()
		optbytes = opts.Bytes()

		coptname = C.CString(options.TypeName())
		defer C.free(unsafe.Pointer(coptname))
	}

	var cargPtrs []*C.struct_ArrowDatum
	datumArgsPtr := C.malloc(C.size_t(unsafe.Sizeof((*C.struct_ArrowDatum)(nil))))
	s := (*reflect.SliceHeader)(unsafe.Pointer(&cargPtrs))
	s.Data = uintptr(datumArgsPtr)
	s.Len, s.Cap = len(args), len(args)
	defer C.free(datumArgsPtr)

	for i, a := range args {
		cargPtrs[i], err = datumToC(a, mem)
		if err != nil {
			return nil, err
		}
		defer freeArrowDatum(cargPtrs[i])
	}

	cfuncname := C.CString(funcName)
	defer C.free(unsafe.Pointer(cfuncname))

	var serialized *C.uint8_t
	if optbytes != nil {
		serialized = (*C.uint8_t)(unsafe.Pointer(&optbytes[0]))
	}

	out := C.arrow_compute_call_function(ec, cfuncname, (**C.struct_ArrowDatum)(datumArgsPtr), C.int(len(args)), coptname, serialized, C.int(len(optbytes)))
	return datumFromC(out, mem)
}

func freeArrowDatum(in *C.struct_ArrowDatum) {
	if in == nil {
		return
	}

	if in.schema != nil {
		schema := cdata.SchemaFromPtr(uintptr(unsafe.Pointer(in.schema)))
		cdata.ReleaseCArrowSchema(schema)
		C.free(unsafe.Pointer(in.schema))
	}

	var arrlist []*cdata.CArrowArray
	if in.num_data > 0 {
		s := (*reflect.SliceHeader)(unsafe.Pointer(&arrlist))
		s.Data = uintptr(unsafe.Pointer(in.data))
		s.Len = int(in.num_data)
		s.Cap = int(in.num_data)

		for _, a := range arrlist {
			cdata.ReleaseCArrowArray(a)
		}

		C.free(unsafe.Pointer(arrlist[0]))
		C.free(unsafe.Pointer(in.data))
	}

	C.free(unsafe.Pointer(in))
}

func datumFromC(in *C.struct_ArrowDatum, mem memory.Allocator) (Datum, error) {
	if DatumKind(in.datum_type) == KindNone {
		return EmptyDatum{}, nil
	}

	var arrlist []*cdata.CArrowArray
	s := (*reflect.SliceHeader)(unsafe.Pointer(&arrlist))
	s.Data = uintptr(unsafe.Pointer(in.data))
	s.Len = int(in.num_data)
	s.Cap = int(in.num_data)

	schema := cdata.SchemaFromPtr(uintptr(unsafe.Pointer(in.schema)))

	var data interface{}

	switch DatumKind(in.datum_type) {
	case KindScalar:
		_, a, err := cdata.ImportCArray(arrlist[0], schema)
		if err != nil {
			return nil, err
		}
		defer a.Release()
		data, err = scalar.GetScalar(a, 0)
		if err != nil {
			return nil, err
		}
	case KindArray:
		var err error
		_, data, err = cdata.ImportCArray(arrlist[0], schema)
		if err != nil {
			return nil, err
		}
	case KindChunked:
		field, err := cdata.ImportCArrowField(schema)
		if err != nil {
			return nil, err
		}

		chunkedList := make([]array.Interface, len(arrlist))
		for i, carr := range arrlist {
			chunkedList[i], err = cdata.ImportCArrayWithType(carr, field.Type)
			if err != nil {
				return nil, err
			}
			defer chunkedList[i].Release()
		}
		data = array.NewChunked(field.Type, chunkedList)
	case KindRecord:
		var err error
		data, err = cdata.ImportCRecordBatch(arrlist[0], schema)
		if err != nil {
			return nil, err
		}
	case KindTable:
		tableSchema, err := cdata.ImportCArrowSchema(schema)
		if err != nil {
			return nil, err
		}

		recordList := make([]array.Record, len(arrlist))
		for i, crec := range arrlist {
			recordList[i], err = cdata.ImportCRecordBatchWithSchema(crec, tableSchema)
			if err != nil {
				return nil, err
			}
			defer recordList[i].Release()
		}

		data = array.NewTableFromRecords(tableSchema, recordList)
	case KindCollection:
		panic("arrow/compute: collection datum from C not implemented yet")
	}

	if r, ok := data.(releasable); ok {
		defer r.Release()
	}

	return NewDatum(data), nil
}

func datumToC(datum Datum, mem memory.Allocator) (out *C.struct_ArrowDatum, err error) {
	var arrs []array.Interface
	switch d := datum.(type) {
	case EmptyDatum:
	case *ScalarDatum:
		arrs = make([]array.Interface, 1)
		arrs[0], err = scalar.MakeArrayFromScalar(d.Value, 1, mem)
	case *ArrayDatum:
		arrs = d.Chunks()
	case *ChunkedDatum:
		arrs = d.Chunks()
	case *RecordDatum:
		arrs = []array.Interface{array.RecordToStructArray(d.Value)}
	case *TableDatum:
		arrs = make([]array.Interface, 0, 1)
		tr := array.NewTableReader(d.Value, -1)
		defer tr.Release()

		for tr.Next() {
			arrs = append(arrs, array.RecordToStructArray(tr.Record()))
		}
	case CollectionDatum:
		panic("arrow/compute: collection datum to C not implemented yet")
	}

	if err != nil {
		return
	}

	out = (*C.struct_ArrowDatum)(C.malloc(C.sizeof_struct_ArrowDatum))
	out.datum_type = C.int(datum.Kind())
	out.num_data = C.int(len(arrs))

	if len(arrs) == 0 {
		return
	}

	scptr := (uintptr)(C.malloc(C.sizeof_struct_ArrowSchema))
	out.schema = (*C.struct_ArrowSchema)(unsafe.Pointer(scptr))

	var allocatedArrs []C.struct_ArrowArray
	s := (*reflect.SliceHeader)(unsafe.Pointer(&allocatedArrs))
	s.Data = uintptr(C.malloc(C.sizeof_struct_ArrowArray * C.size_t(out.num_data)))
	s.Len, s.Cap = len(arrs), len(arrs)

	var arrPtrs []*C.struct_ArrowArray
	s = (*reflect.SliceHeader)(unsafe.Pointer(&arrPtrs))
	s.Data = uintptr(C.malloc(C.size_t(unsafe.Sizeof((*C.struct_ArrowArray)(nil))) * C.size_t(out.num_data)))
	s.Len, s.Cap = len(arrs), len(arrs)

	out.data = (**C.struct_ArrowArray)(unsafe.Pointer(&arrPtrs[0]))

	cdata.ExportArrowArray(arrs[0], cdata.ArrayFromPtr(uintptr(unsafe.Pointer(&allocatedArrs[0]))), cdata.SchemaFromPtr(scptr))
	arrPtrs[0] = &allocatedArrs[0]
	defer arrs[0].Release()

	for i, a := range arrs[1:] {
		defer a.Release()
		cdata.ExportArrowArray(a, cdata.ArrayFromPtr(uintptr(unsafe.Pointer(&allocatedArrs[i+1]))), nil)
		arrPtrs[i+1] = &allocatedArrs[i+1]
	}

	return
}
