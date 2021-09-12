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
	"fmt"
	"reflect"
	"strconv"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/ipc"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/apache/arrow/go/arrow/scalar"
)

type Expression interface {
	IsBound() bool
	IsScalarExpr() bool
	IsNullLiteral() bool
	IsSatisfiable() bool
	FieldRef() *FieldRef
	Descr() ValueDescr
	Type() arrow.DataType
	Equals(Expression) bool
}

type Literal struct {
	Literal Datum
}

func (l *Literal) IsBound() bool      { return l.Type() != nil }
func (l *Literal) IsScalarExpr() bool { return l.Literal.Kind() == KindScalar }

func (l *Literal) Equals(other Expression) bool {
	rhs, ok := other.(*Literal)
	if !ok {
		return false
	}

	return l.Literal.Equals(rhs.Literal)
}

func (l *Literal) IsNullLiteral() bool {
	ad, ok := l.Literal.(ArrayLikeDatum)
	if !ok {
		return true
	}

	return ad.NullN() == ad.Len()
}

func (l *Literal) IsSatisfiable() bool {
	if l.IsNullLiteral() {
		return false
	}

	sc, ok := l.Literal.(*ScalarDatum)
	if ok && sc.Type().ID() == arrow.BOOL {
		return sc.Value.(*scalar.Boolean).Value
	}

	return true
}

func (Literal) FieldRef() *FieldRef { return nil }
func (l *Literal) Descr() ValueDescr {
	ad, ok := l.Literal.(ArrayLikeDatum)
	if ok {
		return ad.Descr()
	}

	return ValueDescr{ShapeAny, nil}
}

func (l *Literal) Type() arrow.DataType {
	ad, ok := l.Literal.(ArrayLikeDatum)
	if ok {
		return ad.Type()
	}

	return nil
}

type Parameter struct {
	ref   *FieldRef
	descr ValueDescr
	index int
}

func (p *Parameter) IsBound() bool        { return p.Type() != nil }
func (p *Parameter) IsScalarExpr() bool   { return p.ref != nil }
func (p *Parameter) IsNullLiteral() bool  { return false }
func (p *Parameter) IsSatisfiable() bool  { return p.Type() == nil || p.Type().ID() != arrow.NULL }
func (p *Parameter) FieldRef() *FieldRef  { return p.ref }
func (p *Parameter) Descr() ValueDescr    { return p.descr }
func (p *Parameter) Type() arrow.DataType { return p.descr.Type }
func (p *Parameter) Equals(other Expression) bool {
	rhs, ok := other.(*Parameter)
	if !ok {
		return false
	}

	return p.ref.Equals(*rhs.ref)
}

type funcopt interface {
	TypeName() string
}

type FunctionOptions struct {
	opt funcopt
}

func (f FunctionOptions) ToStructScalar(mem memory.Allocator) (*scalar.Struct, error) {
	st, err := scalar.ToScalar(f.opt, mem)
	if err != nil {
		return nil, err
	}
	return st.(*scalar.Struct), nil
}

func (f FunctionOptions) Equals(rhs *FunctionOptions) bool { return reflect.DeepEqual(f.opt, rhs.opt) }

type Call struct {
	funcName string
	args     []Expression
	descr    ValueDescr
	options  *FunctionOptions
}

func (c *Call) IsNullLiteral() bool  { return false }
func (c *Call) FieldRef() *FieldRef  { return nil }
func (c *Call) Descr() ValueDescr    { return c.descr }
func (c *Call) Type() arrow.DataType { return c.descr.Type }
func (c *Call) IsSatisfiable() bool  { return c.Type() == nil || c.Type().ID() != arrow.NULL }

func (c *Call) IsBound() bool {
	if c.Type() == nil {
		return false
	}

	for _, arg := range c.args {
		if !arg.IsBound() {
			return false
		}
	}
	return true
}

func (c *Call) IsScalarExpr() bool {
	for _, arg := range c.args {
		if !arg.IsScalarExpr() {
			return false
		}
	}

	// check function if it is scalar or not

	return false
}

func (c *Call) Equals(other Expression) bool {
	rhs, ok := other.(*Call)
	if !ok {
		return false
	}

	if c.funcName != rhs.funcName {
		return false
	}

	if len(c.args) != len(rhs.args) {
		return false
	}

	for i := range c.args {
		if !c.args[i].Equals(rhs.args[i]) {
			return false
		}
	}

	return c.options.Equals(rhs.options)
}

func NewLiteral(arg interface{}) Expression {
	return &Literal{NewDatum(arg)}
}

func NewRef(ref FieldRef) Expression {
	return &Parameter{ref: &ref, index: -1}
}

func NewFieldRef(field string) Expression {
	return NewRef(NewFieldNameRef(field))
}

func NewCall(name string, args []Expression, opts *FunctionOptions) Expression {
	return &Call{funcName: name, args: args, options: opts}
}

func FieldsInExpression(expr Expression) []FieldRef {
	switch expr := expr.(type) {
	case *Literal:
		return []FieldRef{}
	case *Parameter:
		return []FieldRef{*expr.ref}
	case *Call:
		out := make([]FieldRef, 0)
		for _, arg := range expr.args {
			argFields := FieldsInExpression(arg)
			out = append(out, argFields...)
		}
		return out
	}

	return nil
}

func ExpressionHasFieldRefs(expr Expression) bool {
	switch expr := expr.(type) {
	case *Literal:
		return false
	case *Parameter:
		return true
	case *Call:
		for _, arg := range expr.args {
			if ExpressionHasFieldRefs(arg) {
				return true
			}
		}
	}

	return false
}

type MakeStructOptions struct {
	FieldNames       []string          `compute:"field_names"`
	FieldNullability []bool            `compute:"field_nullability"`
	FieldMetadata    []*arrow.Metadata `compute:"field_metadata"`
}

func (MakeStructOptions) TypeName() string { return "MakeStructOptions" }

type NullOptions struct {
	NanIsNull bool `compute:"nan_is_null"`
}

func (NullOptions) TypeName() string { return "NullOptions" }

type StrptimeOptions struct {
	Format string         `compute:"format"`
	Unit   arrow.TimeUnit `compute:"unit"`
}

func (StrptimeOptions) TypeName() string { return "StrptimeOptions" }

func Project(values []Expression, names []string) Expression {
	nulls := make([]bool, len(names))
	for i := range nulls {
		nulls[i] = true
	}
	meta := make([]*arrow.Metadata, len(names))
	return NewCall("make_struct", values,
		&FunctionOptions{&MakeStructOptions{FieldNames: names, FieldNullability: nulls, FieldMetadata: meta}})
}

func Equal(lhs, rhs Expression) Expression {
	return NewCall("equal", []Expression{lhs, rhs}, nil)
}

func NotEqual(lhs, rhs Expression) Expression {
	return NewCall("not_equal", []Expression{lhs, rhs}, nil)
}

func Less(lhs, rhs Expression) Expression {
	return NewCall("less", []Expression{lhs, rhs}, nil)
}

func LessEqual(lhs, rhs Expression) Expression {
	return NewCall("less_equal", []Expression{lhs, rhs}, nil)
}

func Greater(lhs, rhs Expression) Expression {
	return NewCall("greater", []Expression{lhs, rhs}, nil)
}

func GreaterEqual(lhs, rhs Expression) Expression {
	return NewCall("greater_equal", []Expression{lhs, rhs}, nil)
}

func IsNull(lhs Expression, nanIsNull bool) Expression {
	return NewCall("less", []Expression{lhs}, &FunctionOptions{&NullOptions{nanIsNull}})
}

func IsValid(lhs Expression) Expression {
	return NewCall("is_valid", []Expression{lhs}, nil)
}

type binop func(lhs, rhs Expression) Expression

func foldLeft(op binop, args ...Expression) Expression {
	switch len(args) {
	case 0:
		return nil
	case 1:
		return args[0]
	}

	folded := args[0]
	for _, a := range args[1:] {
		folded = op(folded, a)
	}
	return folded
}

func And(lhs, rhs Expression) Expression {
	return NewCall("and_kleene", []Expression{lhs, rhs}, nil)
}

func AndList(ops ...Expression) Expression {
	folded := foldLeft(And, ops...)
	if folded != nil {
		return folded
	}
	return NewLiteral(true)
}

func Or(lhs, rhs Expression) Expression {
	return NewCall("or_kleene", []Expression{lhs, rhs}, nil)
}

func OrList(ops ...Expression) Expression {
	folded := foldLeft(Or, ops...)
	if folded != nil {
		return folded
	}
	return NewLiteral(false)
}

func Not(expr Expression) Expression {
	return NewCall("invert", []Expression{expr}, nil)
}

// SerializeExpr serializes expressions by converting them to Metadata and
// storing this in the schema of a Record. Embedded arrays and scalars are
// stored in its columns. Finally the record is written as an IPC file
func SerializeExpr(expr Expression, mem memory.Allocator) *memory.Buffer {
	var (
		cols      []array.Interface
		metaKey   []string
		metaValue []string
		visit     func(Expression)
	)

	addScalar := func(s scalar.Scalar) string {
		ret := len(cols)
		arr, err := scalar.MakeArrayFromScalar(s, 1, mem)
		if err != nil {
			panic(err)
		}
		cols = append(cols, arr)
		return strconv.Itoa(ret)
	}

	visit = func(e Expression) {
		switch e := e.(type) {
		case *Literal:
			if !e.IsScalarExpr() {
				panic("not implemented: serialization of non-scalar literal")
			}

			metaKey = append(metaKey, "literal")
			metaValue = append(metaValue, addScalar(e.Literal.(*ScalarDatum).Value))
		case *Parameter:
			if e.ref.Name() == "" {
				panic("not implemented: serialization of non-name field_ref")
			}

			metaKey = append(metaKey, "field_ref")
			metaValue = append(metaValue, e.ref.Name())
		case *Call:
			metaKey = append(metaKey, "call")
			metaValue = append(metaValue, e.funcName)

			for _, arg := range e.args {
				visit(arg)
			}

			if e.options != nil {
				st, err := e.options.ToStructScalar(mem)
				if err != nil {
					panic(err)
				}
				metaKey = append(metaKey, "options")
				metaValue = append(metaValue, addScalar(st))

				for _, f := range st.Value {
					switch s := f.(type) {
					case scalar.ListScalar:
						defer func(sc scalar.ListScalar) {
							if sc.List().DataType().ID() == arrow.MAP {
								d := sc.List().(*array.Map).ListValues().Data()
								sc.List().Release()
								fmt.Printf("%+v\n", d)
							} else {
								sc.List().Release()
							}
						}(s)
					case scalar.BinaryScalar:
						defer s.Release()
					}
				}
			}

			metaKey = append(metaKey, "end")
			metaValue = append(metaValue, e.funcName)
		}
	}

	visit(expr)
	fields := make([]arrow.Field, len(cols))
	for i, c := range cols {
		fields[i].Type = c.DataType()
		defer c.Release()
	}

	metadata := arrow.NewMetadata(metaKey, metaValue)
	rec := array.NewRecord(arrow.NewSchema(fields, &metadata), cols, 1)
	defer rec.Release()

	buf := &bufferWriteSeeker{mem: mem}
	wr, err := ipc.NewFileWriter(buf, ipc.WithSchema(rec.Schema()), ipc.WithAllocator(mem))
	if err != nil {
		panic(err)
	}

	wr.Write(rec)
	wr.Close()
	return buf.buf
}
