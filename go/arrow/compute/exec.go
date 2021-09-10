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

package compute

import (
	"context"
	"unsafe"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/cdata"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/apache/arrow/go/arrow/scalar"
	"golang.org/x/xerrors"
)

// #cgo pkg-config: arrow
// #include "arrow/c/helpers.h"
// #include "arrow/compute/c/abi.h"
// #include <stdlib.h>
//
// struct ArrowComputeInputOutput get_io() {
//  return (struct ArrowComputeInputOutput) {
//		.data = malloc(sizeof(struct ArrowArray)),
//      .schema = malloc(sizeof(struct ArrowSchema)),
//  };
// }
//
// void release_io(struct ArrowComputeInputOutput* io) {
//   free(io->data);
//   io->data = NULL;
//   free(io->schema);
//   io->schema = NULL;
// }
import "C"

type ExecBatch struct {
	values []Datum
	length int64
}

func NewExecBatch(fullSchema *arrow.Schema, partial Datum) (out ExecBatch, err error) {
	if partial.Kind() == KindRecordBatch {
		out.values = make([]Datum, 0, len(fullSchema.Fields()))

		batch := partial.(*RecordDatum)
		out.length = batch.Len()
		for _, f := range fullSchema.Fields() {
			col, err := NewFieldNameRef(f.Name).GetOneColumnOrNone(batch.Value)
			if err != nil {
				return out, err
			}

			if col != nil {
				if !arrow.TypeEqual(col.DataType(), f.Type) {
					// referenced field was present but didn't have the expected type
					// for now return an error, possibly cast in the future
					return out, xerrors.Errorf("expected type %s for %s, got %s", f.Type, f.Name, col.DataType())
				}
				out.values = append(out.values, NewDatum(col))
			} else {
				out.values = append(out.values, NewDatum(scalar.MakeNullScalar(f.Type)))
			}
		}
		return
	}

	err = xerrors.Errorf("not implemented NewExecBatch from %s", partial)
	return
}

type execCtxKey struct{}

func ExecContext(ctx context.Context) context.Context {
	ec := C.arrow_compute_default_context()
	return context.WithValue(ctx, execCtxKey{}, ec)
}

func ReleaseContext(ctx context.Context) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	C.arrow_compute_release_context(ec)
}

func ExecuteScalarExpr(ctx context.Context, rb array.Record, expr Expression) (out Datum, err error) {
	ec := ctx.Value(execCtxKey{}).(C.ExecContext)
	input := C.get_io()
	defer C.release_io(&input)

	cdata.ExportArrowRecordBatch(rb, (*cdata.CArrowArray)(unsafe.Pointer(input.data)), (*cdata.CArrowSchema)(unsafe.Pointer(input.schema)))
	defer C.ArrowArrayRelease(input.data)
	defer C.ArrowSchemaRelease(input.schema)

	output := C.get_io()
	defer C.ArrowArrayRelease(output.data)
	defer C.ArrowSchemaRelease(output.schema)
	defer C.release_io(&output)

	buf := SerializeExpr(expr, memory.DefaultAllocator)
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
