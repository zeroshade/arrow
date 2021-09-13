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
	defer cdata.CArrowArrayRelease(convCArr(output.data))
	defer cdata.CArrowSchemaRelease(convCSchema(output.schema))
	defer C.release_io(&output)

	buf := SerializeExpr(expr, mem)
	defer buf.Release()
	if ec := C.arrow_compute_execute_scalar_expr(ec, &input, (*C.uint8_t)(unsafe.Pointer(&buf.Bytes()[0])), C.int(buf.Len()), &output); ec != 0 {
		return nil, xerrors.Errorf("got errorcode: %d", ec)
	}

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
