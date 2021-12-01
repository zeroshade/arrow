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

//go:build cgo && ccalloc
// +build cgo,ccalloc

package compute_test

import (
	"context"
	"runtime"
	"testing"

	"github.com/apache/arrow/go/v7/arrow"
	"github.com/apache/arrow/go/v7/arrow/array"
	"github.com/apache/arrow/go/v7/arrow/compute"
	"github.com/apache/arrow/go/v7/arrow/memory"
	"github.com/stretchr/testify/suite"
)

type ComputeTestSuite struct {
	suite.Suite

	alloc *memory.CgoArrowAllocator
	mem   *memory.CheckedAllocator
}

func (c *ComputeTestSuite) SetupSuite() {
	c.alloc = memory.NewCgoArrowAllocator()
	c.mem = memory.NewCheckedAllocator(c.alloc)
}

func (c *ComputeTestSuite) SetupTest() {

}

func (c *ComputeTestSuite) TearDownTest() {
	runtime.GC()
	c.mem.AssertSize(c.T(), 0)
	c.alloc.AssertSize(c.T(), 0)
}

func (c *ComputeTestSuite) simpleRecord() array.Record {
	bldr := array.NewRecordBuilder(c.mem, arrow.NewSchema([]arrow.Field{{Name: "a", Type: arrow.PrimitiveTypes.Float64}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	return bldr.NewRecord()
}

func (c *ComputeTestSuite) twoColRecord() array.Record {
	bldr := array.NewRecordBuilder(c.mem, arrow.NewSchema(
		[]arrow.Field{
			{Name: "a", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
			{Name: "b", Type: arrow.PrimitiveTypes.Int32, Nullable: true}}, nil))
	defer bldr.Release()

	abldr := bldr.Field(0).(*array.Float64Builder)
	abldr.AppendValues([]float64{6.125, 0, -1}, nil)

	bbldr := bldr.Field(1).(*array.Int32Builder)
	bbldr.AppendValues([]int32{10, 20, 30}, nil)

	return bldr.NewRecord()
}

func (c *ComputeTestSuite) TestScalarExecute() {
	rb := c.simpleRecord()
	defer rb.Release()

	expr, err := compute.NewFieldRef("a").Bind(context.TODO(), c.mem, rb.Schema())
	c.NoError(err)
	defer expr.Release()

	input := compute.NewDatum(rb)
	defer input.Release()

	output, err := compute.ExecuteScalarExpression(context.TODO(), expr, c.mem, rb.Schema(), input)
	c.NoError(err)

	c.IsType((*compute.ArrayDatum)(nil), output)
	expected := compute.NewDatum(rb.Column(0))
	defer expected.Release()

	c.True(expected.Equals(output))
}

func (c *ComputeTestSuite) TestScalarCallFunc() {
	rb := c.simpleRecord()
	defer rb.Release()

	expr, err := compute.NewCall("add", []compute.Expression{compute.NewFieldRef("a"), compute.NewLiteral(3.5)}, nil).
		Bind(context.TODO(), c.mem, rb.Schema())
	c.NoError(err)
	defer expr.Release()

	input := compute.NewDatum(rb)
	defer input.Release()

	out, err := compute.ExecuteScalarExpression(context.TODO(), expr, c.mem, rb.Schema(), input)
	c.NoError(err)

	fltbldr := array.NewFloat64Builder(c.mem)
	defer fltbldr.Release()

	fltbldr.AppendValues([]float64{6.125 + 3.5, 0 + 3.5, -1 + 3.5}, nil)
	expected := fltbldr.NewArray()
	defer expected.Release()

	arr := out.(*compute.ArrayDatum).MakeArray()
	defer arr.Release()

	c.True(array.ArrayApproxEqual(arr, expected))
}

func (c *ComputeTestSuite) TestProjectCallExpr() {
	rb := c.twoColRecord()
	defer rb.Release()

	ctx := compute.WithExecCtx(context.Background())
	defer compute.ReleaseExecContext(ctx)
	expr, err := compute.Project([]compute.Expression{compute.NewFieldRef("a"), compute.NewFieldRef("b")}, []string{"a", "b"}).
		Bind(ctx, c.mem, rb.Schema())
	c.NoError(err)
	defer expr.Release()

	input := compute.NewDatum(rb)
	defer input.Release()

	out, err := compute.ExecuteScalarExpression(ctx, expr, c.mem, rb.Schema(), input)
	c.NoError(err)
	defer out.Release()

	expected := array.RecordToStructArray(rb)
	defer expected.Release()

	exdatum := compute.NewDatum(expected)
	defer exdatum.Release()

	c.Truef(exdatum.Equals(out), "expected: %s\ngot: %s\n", expected, out)
}

func (c *ComputeTestSuite) TestProjectAddExpr() {
	rb := c.twoColRecord()
	defer rb.Release()

	expr, err := compute.Project([]compute.Expression{compute.NewCall("add", []compute.Expression{compute.NewFieldRef("a"), compute.NewLiteral(3.5)}, nil)}, []string{"a + 3.5"}).
		Bind(context.TODO(), c.mem, rb.Schema())
	c.NoError(err)
	defer expr.Release()

	input := compute.NewDatum(rb)
	defer input.Release()

	out, err := compute.ExecuteScalarExpression(context.TODO(), expr, c.mem, rb.Schema(), input)
	c.NoError(err)
	defer out.Release()

	rb2bldr := array.NewStructBuilder(c.mem,
		arrow.StructOf(arrow.Field{Name: "a + 3.5", Type: arrow.PrimitiveTypes.Float64, Nullable: true}))
	defer rb2bldr.Release()

	cola := rb.Column(0).(*array.Float64)

	a2bldr := rb2bldr.FieldBuilder(0).(*array.Float64Builder)
	for _, v := range cola.Float64Values() {
		rb2bldr.Append(true)
		a2bldr.Append(v + 3.5)
	}

	rb2 := rb2bldr.NewStructArray()
	defer rb2.Release()

	expected := compute.NewDatum(rb2)
	defer expected.Release()

	c.Truef(expected.Equals(out), "expected: %s\ngot: %s\n", expected, out)
}

func (c *ComputeTestSuite) TestStrptimeCallExpr() {
	bldr := array.NewStructBuilder(c.mem, arrow.StructOf(arrow.Field{Name: "a", Type: arrow.BinaryTypes.String, Nullable: true}))
	defer bldr.Release()

	bldr.AppendValues([]bool{true, true, true})
	abldr := bldr.FieldBuilder(0).(*array.StringBuilder)
	abldr.AppendValues([]string{"5/1/2020", "", "12/11/1900"}, []bool{true, false, true})

	arr := bldr.NewStructArray()
	defer arr.Release()

	input := compute.NewDatum(arr)
	defer input.Release()

	fullSchema := arrow.NewSchema(arr.DataType().(*arrow.StructType).Fields(), nil)
	expr, err := compute.NewCall("strptime", []compute.Expression{compute.NewFieldRef("a")}, &compute.StrptimeOptions{Format: "%m/%d/%Y", Unit: arrow.Millisecond}).
		Bind(context.TODO(), c.mem, fullSchema)
	c.NoError(err)
	defer expr.Release()

	out, err := compute.ExecuteScalarExpression(context.Background(), expr, c.mem, fullSchema, input)
	c.NoError(err)
	defer out.Release()

	result := out.(*compute.ArrayDatum).MakeArray().(*array.Timestamp)
	defer result.Release()

	c.True(arrow.TypeEqual(arrow.FixedWidthTypes.Timestamp_ms, result.DataType()))
	c.Equal(arrow.Timestamp(1588291200000), result.Value(0))
	c.True(result.IsNull(1))
	c.Equal(arrow.Timestamp(-2179267200000), result.Value(2))
}

func (c *ComputeTestSuite) TestCallFunction() {
	bldr := array.NewInt32Builder(c.mem)
	defer bldr.Release()

	bldr.AppendValues([]int32{1, 2, 3, 4, 5, 6},
		[]bool{true, true, false, true, false, true})
	arr := bldr.NewInt32Array()
	defer arr.Release()

	left := compute.NewDatum(arr)
	defer left.Release()

	output, err := compute.CallFunction(context.Background(), c.mem, "add",
		[]compute.Datum{left, compute.NewDatum(int32(5))},
		&compute.ArithmeticOptions{false})
	c.NoError(err)
	defer output.Release()

	bldr.AppendValues([]int32{6, 7, 0, 9, 0, 11},
		[]bool{true, true, false, true, false, true})
	expected := bldr.NewInt32Array()
	defer expected.Release()

	outarr := output.(*compute.ArrayDatum).MakeArray()
	defer outarr.Release()

	c.Truef(array.ArrayEqual(expected, outarr), "expected: %s\ngot: %s\n", expected, outarr)
}

func TestCompute(t *testing.T) {
	suite.Run(t, new(ComputeTestSuite))
}

var boringSchema = arrow.NewSchema([]arrow.Field{
	{Name: "bool", Type: arrow.FixedWidthTypes.Boolean, Nullable: true},
	{Name: "i8", Type: arrow.PrimitiveTypes.Int8, Nullable: true},
	{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
	{Name: "i32_req", Type: arrow.PrimitiveTypes.Int8, Nullable: false},
	{Name: "u32", Type: arrow.PrimitiveTypes.Uint32, Nullable: true},
	{Name: "i64", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
	{Name: "f32", Type: arrow.PrimitiveTypes.Float32, Nullable: true},
	{Name: "f32_req", Type: arrow.PrimitiveTypes.Float32, Nullable: false},
	{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	{Name: "date64", Type: arrow.PrimitiveTypes.Date64, Nullable: true},
	{Name: "str", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "ts_ns", Type: arrow.FixedWidthTypes.Timestamp_ns, Nullable: true},
}, nil)

type BindTestSuite struct {
	suite.Suite

	ctx   context.Context
	alloc *memory.CgoArrowAllocator
	mem   *memory.CheckedAllocator
}

func (b *BindTestSuite) SetupSuite() {
	b.ctx = compute.WithExecCtx(context.Background())
	b.alloc = memory.NewCgoArrowAllocator()
	b.mem = memory.NewCheckedAllocator(b.alloc)
}

func (b *BindTestSuite) TearDownSuite() {
	compute.ReleaseExecContext(b.ctx)
}

func (b *BindTestSuite) SetupTest() {

}

func (b *BindTestSuite) TearDownTest() {
	runtime.GC()
	b.mem.AssertSize(b.T(), 0)
	b.alloc.AssertSize(b.T(), 0)
}

func (b *BindTestSuite) expectBindsTo(expr, expected compute.Expression, boundOut *compute.Expression) {
	if expected == nil {
		expected = expr
	}

	bound, err := expr.Bind(b.ctx, b.mem, boringSchema)
	b.NoError(err)
	b.True(bound.IsBound())

	expected, err = expected.Bind(b.ctx, b.mem, boringSchema)
	b.NoError(err)
	b.Truef(bound.Equals(expected), " unbound: %s, expected: %s", expr, expected)
	defer expected.Release()

	if boundOut != nil {
		*boundOut = bound
	} else {
		bound.Release()
	}
}

func (b *BindTestSuite) TestBindLiteral() {
	bldr := array.NewInt32Builder(b.mem)
	defer bldr.Release()
	bldr.AppendValues([]int32{1, 2, 3}, nil)
	arr := bldr.NewInt32Array()
	defer arr.Release()

	for _, dat := range []compute.Datum{compute.NewDatum(3), compute.NewDatum(3.5), compute.NewDatum(arr)} {
		defer dat.Release()

		expr := &compute.Literal{Literal: dat}
		b.Equal(expr.Descr(), dat.(compute.ArrayLikeDatum).Descr())
		b.True(expr.IsBound())
	}
}

func (b *BindTestSuite) TestBindFieldRef() {
	expr := compute.NewFieldRef("alpha")
	b.Equal(expr.Descr(), compute.ValueDescr{})
	b.False(expr.IsBound())

	b.expectBindsTo(compute.NewFieldRef("i32"), nil, &expr)
	defer expr.Release()
	b.Equal(expr.Descr(), compute.ValueDescr{Shape: compute.ShapeArray, Type: arrow.PrimitiveTypes.Int32})

	// if field is not found, returns an error
	_, err := compute.NewFieldRef("no such field").Bind(b.ctx, b.mem, boringSchema)
	b.Error(err)

	// referencing nested fields is not supported yet
	_, err = compute.NewRef(compute.FieldRefList("a", "b")).Bind(b.ctx, b.mem, arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.StructOf(arrow.Field{Name: "b", Type: arrow.PrimitiveTypes.Int32})},
	}, nil))
	b.Error(err)
}

func (b *BindTestSuite) TestBindCall() {
	expr := compute.NewCall("add", []compute.Expression{compute.NewFieldRef("i32"), compute.NewFieldRef("i32_req")}, nil)
	b.False(expr.IsBound())

	b.expectBindsTo(expr, nil, &expr)
	defer expr.Release()
	b.Equal(expr.Descr(), compute.ValueDescr{Shape: compute.ShapeArray, Type: arrow.PrimitiveTypes.Int32})

	b.expectBindsTo(compute.NewCall("add", []compute.Expression{compute.NewFieldRef("f32"), compute.NewLiteral(3)}, nil),
		compute.NewCall("add", []compute.Expression{compute.NewFieldRef("f32"), compute.NewLiteral(float32(3.0))}, nil), nil)

	b.expectBindsTo(compute.NewCall("add", []compute.Expression{compute.NewFieldRef("i32"), compute.NewLiteral(float32(3.5))}, nil),
		compute.NewCall("add", []compute.Expression{compute.Cast(compute.NewFieldRef("i32"), arrow.PrimitiveTypes.Float32), compute.NewLiteral(float32(3.5))}, nil), nil)
}

func (b *BindTestSuite) TestBindNestedCall() {
	expr := compute.NewCall("add", []compute.Expression{compute.NewFieldRef("a"),
		compute.NewCall("subtract", []compute.Expression{
			compute.NewCall("multiply", []compute.Expression{compute.NewFieldRef("b"), compute.NewFieldRef("c")}, nil),
			compute.NewFieldRef("d"),
		}, nil)}, nil)

	b.False(expr.IsBound())
	var err error
	expr, err = expr.Bind(b.ctx, b.mem, arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "b", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "c", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "d", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
	}, nil))
	b.NoError(err)
	defer expr.Release()

	b.Equal(compute.ValueDescr{compute.ShapeArray, arrow.PrimitiveTypes.Int32}, expr.Descr())
	b.True(expr.IsBound())
}

func TestBind(t *testing.T) {
	suite.Run(t, new(BindTestSuite))
}
