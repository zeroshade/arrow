// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// +build cgo
// +build ccalloc

package compute

import (
	"context"
	"runtime"
	"testing"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/internal/testing/types"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/stretchr/testify/assert"
)

func TestScalarCompute(t *testing.T) {
	alloc := memory.NewCgoArrowAllocator()
	defer alloc.AssertSize(t, 0)

	mem := memory.NewCheckedAllocator(alloc)
	defer mem.AssertSize(t, 0)

	defer func() {
		runtime.GC()
	}()

	bldr := array.NewRecordBuilder(mem, arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Float64}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	rb := bldr.NewRecord()
	defer rb.Release()

	expr := NewRef(NewFieldNameRef("a"))
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)

	out, err := ExecuteScalarExpr(ctx, alloc, rb, expr)
	assert.NoError(t, err)
	defer out.Release()

	exp := NewDatum(rb.Column(0))
	defer exp.Release()

	assert.IsType(t, (*ArrayDatum)(nil), out)
	assert.True(t, out.Equals(exp))
}

func TestScalarCallFunc(t *testing.T) {
	alloc := memory.NewCgoArrowAllocator()
	defer alloc.AssertSize(t, 0)

	mem := memory.NewCheckedAllocator(alloc)
	defer mem.AssertSize(t, 0)

	defer func() {
		runtime.GC()
	}()

	bldr := array.NewRecordBuilder(mem, arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Float64}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	rb := bldr.NewRecord()
	defer rb.Release()

	expr := NewCall("add", []Expression{NewFieldRef("a"), NewLiteral(3.5)}, nil)
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)

	out, err := ExecuteScalarExpr(ctx, mem, rb, expr)
	assert.NoError(t, err)
	defer out.Release()

	fltbldr := array.NewFloat64Builder(mem)
	defer fltbldr.Release()

	fltbldr.AppendValues([]float64{6.125 + 3.5, 0 + 3.5, -1 + 3.5}, nil)
	expected := fltbldr.NewArray()
	defer expected.Release()

	arr := out.(*ArrayDatum).MakeArray()
	defer arr.Release()
	assert.True(t, array.ArrayApproxEqual(arr, expected))
}

func TestProjectCallExpr(t *testing.T) {
	alloc := memory.NewCgoArrowAllocator()
	defer alloc.AssertSize(t, 0)

	mem := memory.NewCheckedAllocator(alloc)
	defer mem.AssertSize(t, 0)

	defer func() {
		runtime.GC()
	}()

	bldr := array.NewRecordBuilder(mem, arrow.NewSchema(
		[]arrow.Field{
			{Name: "a", Type: arrow.PrimitiveTypes.Float64},
			{Name: "b", Type: arrow.PrimitiveTypes.Int32}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	bbldr := bldr.Field(1).(*array.Int32Builder)
	bbldr.AppendValues([]int32{10, 20, 30}, nil)

	rb := bldr.NewRecord()
	defer rb.Release()

	expr := Project([]Expression{NewFieldRef("a"), NewFieldRef("b")}, []string{"a", "b"})
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)

	out, err := ExecuteScalarExpr(ctx, mem, rb, expr)
	assert.NoError(t, err)
	defer out.(*RecordDatum).Value.Release()

	assert.IsType(t, (*RecordDatum)(nil), out)
	assert.True(t, array.RecordEqual(rb, out.(*RecordDatum).Value))
}

func TestProjectAddExpr(t *testing.T) {
	alloc := memory.NewCgoArrowAllocator()
	defer alloc.AssertSize(t, 0)

	mem := memory.NewCheckedAllocator(alloc)
	defer mem.AssertSize(t, 0)

	defer func() {
		runtime.GC()
	}()

	bldr := array.NewRecordBuilder(mem, arrow.NewSchema(
		[]arrow.Field{
			{Name: "a", Type: arrow.PrimitiveTypes.Float64},
			{Name: "b", Type: arrow.PrimitiveTypes.Int32}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	bbldr := bldr.Field(1).(*array.Int32Builder)
	bbldr.AppendValues([]int32{10, 20, 30}, nil)

	rb := bldr.NewRecord()
	defer rb.Release()

	expr := Project([]Expression{NewCall("add", []Expression{NewFieldRef("a"), NewLiteral(3.5)}, nil)}, []string{"a + 3.5"})
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)

	out, err := ExecuteScalarExpr(ctx, mem, rb, expr)
	assert.NoError(t, err)
	assert.IsType(t, (*RecordDatum)(nil), out)
	defer out.Release()

	rb2Bldr := array.NewRecordBuilder(mem,
		arrow.NewSchema([]arrow.Field{{Name: "a + 3.5", Type: arrow.PrimitiveTypes.Float64, Nullable: true}}, nil))
	defer rb2Bldr.Release()

	cola := rb.Column(0).(*array.Float64)
	a2bldr := rb2Bldr.Field(0).(*array.Float64Builder)
	for _, v := range cola.Float64Values() {
		a2bldr.Append(v + 3.5)
	}

	rb2 := rb2Bldr.NewRecord()
	defer rb2.Release()
	assert.True(t, array.RecordEqual(rb2, out.(*RecordDatum).Value))
}

func TestStrptimeCallExpr(t *testing.T) {
	alloc := memory.NewCgoArrowAllocator()
	defer alloc.AssertSize(t, 0)

	mem := memory.NewCheckedAllocator(alloc)
	defer mem.AssertSize(t, 0)

	defer func() {
		runtime.GC()
	}()

	bldr := array.NewRecordBuilder(mem, arrow.NewSchema(
		[]arrow.Field{{Name: "a", Type: arrow.BinaryTypes.String, Nullable: true}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.StringBuilder)
	abldr.AppendValues([]string{"5/1/2020", "", "12/11/1900"}, []bool{true, false, true})

	rb := bldr.NewRecord()
	defer rb.Release()

	expr := NewCall("strptime", []Expression{NewFieldRef("a")}, &FunctionOptions{StrptimeOptions{Format: "%m/%d/%Y", Unit: arrow.Millisecond}})
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)

	out, err := ExecuteScalarExpr(ctx, mem, rb, expr)
	assert.NoError(t, err)
	defer out.Release()

	arr := out.(*ArrayDatum).MakeArray().(*array.Timestamp)
	defer arr.Release()

	assert.True(t, arrow.TypeEqual(arrow.FixedWidthTypes.Timestamp_ms, arr.DataType()))
	assert.Equal(t, arrow.Timestamp(1588291200000), arr.Value(0))
	assert.True(t, arr.IsNull(1))
	assert.Equal(t, arrow.Timestamp(-2179267200000), arr.Value(2))
}

func TestBindExpression(t *testing.T) {
	expr := Project([]Expression{NewFieldRef("i32"), NewFieldRef("f64")}, []string{"i32", "f64"})

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32},
		{Name: "i64", Type: arrow.PrimitiveTypes.Int64},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil)

	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)
	bound := BindExpression(ctx, memory.DefaultAllocator, expr, schema)
	dt := bound.Type()
	i32 := schema.Field(0)
	i32.Nullable = true
	exp := arrow.StructOf(i32, schema.Field(2))
	assert.True(t, arrow.TypeEqual(dt, exp), dt.(*arrow.StructType).String(), exp.String())
}

func TestFilter(t *testing.T) {
	boringSchema := arrow.NewSchema([]arrow.Field{
		{Name: "i64", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
		{Name: "f32", Type: arrow.PrimitiveTypes.Float32, Nullable: true},
	}, nil)

	input, err := types.RecordFromJSON(boringSchema, []byte(`[
		{"i64": 0, "f32": 0.1},
		{"i64": 0, "f32": 0.3},
		{"i64": 1, "f32": 0.5},
		{"i64": 2, "f32": 0.1},
		{"i64": 0, "f32": 0.1},
		{"i64": 0, "f32": 0.4},
		{"i64": 0, "f32": 1.0}
	]`))
	assert.NoError(t, err)

	expr := Greater(NewFieldRef("f32"), NewLiteral(float32(0.3)))
	ctx := ExecContext(context.Background())
	defer ReleaseContext(ctx)
	bound := BindExpression(ctx, memory.DefaultAllocator, expr, boringSchema)

	mask, err := ExecuteScalarExprWithSchema(ctx, memory.DefaultAllocator, boringSchema, input, bound)
	assert.NoError(t, err)

	out, err := Filter(ctx, memory.DefaultAllocator, NewDatum(input), mask, FilterOptions{})
	assert.NoError(t, err)

	expected, _ := types.RecordFromJSON(boringSchema, []byte(`[
		{"i64": 1, "f32": 0.5},
		{"i64": 0, "f32": 0.4},
		{"i64": 0, "f32": 1.0}
	]`))

	assert.True(t, out.Equals(NewDatum(expected)))
}
