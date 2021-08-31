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
	"strings"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/scalar"
)

//go:generate go run golang.org/x/tools/cmd/stringer -type=ValueShape -linecomment
//go:generate go run golang.org/x/tools/cmd/stringer -type=DatumKind -linecomment

type ValueShape int8

const (
	ShapeAny    ValueShape = iota // any
	ShapeArray                    // array
	ShapeScalar                   // scalar
)

type ValueDescr struct {
	Shape ValueShape
	Type  arrow.DataType
}

func (v *ValueDescr) String() string {
	return fmt.Sprintf("%s [%s]", v.Shape, v.Type)
}

func NewDescrAny(typ arrow.DataType) ValueDescr    { return ValueDescr{ShapeAny, typ} }
func NewDescrArray(typ arrow.DataType) ValueDescr  { return ValueDescr{ShapeArray, typ} }
func NewDescrScalar(typ arrow.DataType) ValueDescr { return ValueDescr{ShapeScalar, typ} }

type DatumKind int

const (
	KindNone         DatumKind = iota // none
	KindScalar                        // scalar
	KindArray                         // array
	KindChunkedArray                  // chunked_array
	KindRecordBatch                   // record_batch
	KindTable                         // table
	KindCollection                    // collection
)

const UnknownLength int64 = -1

type Datum interface {
	fmt.Stringer
	Kind() DatumKind
	Len() int64
	Equals(Datum) bool
}

type ArrayLikeDatum interface {
	Datum
	Shape() ValueShape
	Descr() ValueDescr
	NullN() int64
	Type() arrow.DataType
	Chunks() []array.Interface
}

type RecordLikeDatum interface {
	Datum
	Schema() *arrow.Schema
}

func NewDatum(value interface{}) Datum {
	switch v := value.(type) {
	case array.Interface:
		v.Data().Retain()
		return &ArrayDatum{v.Data()}
	case *array.Chunked:
		v.Retain()
		return &ChunkedDatum{v}
	case array.Record:
		v.Retain()
		return &RecordDatum{v}
	case array.Table:
		v.Retain()
		return &TableDatum{v}
	case []Datum:
		return CollectionDatum(v)
	case scalar.Scalar:
		return &ScalarDatum{v}
	default:
		return &ScalarDatum{scalar.MakeScalar(value)}
	}
}

type EmptyDatum struct{}

func (EmptyDatum) String() string  { return "nullptr" }
func (EmptyDatum) Kind() DatumKind { return KindNone }
func (EmptyDatum) Len() int64      { return UnknownLength }
func (EmptyDatum) Equals(other Datum) bool {
	_, ok := other.(EmptyDatum)
	return ok
}

type ScalarDatum struct {
	Value scalar.Scalar
}

func (ScalarDatum) Kind() DatumKind              { return KindScalar }
func (ScalarDatum) Shape() ValueShape            { return ShapeScalar }
func (ScalarDatum) Len() int64                   { return 1 }
func (s *ScalarDatum) Chunks() []array.Interface { return nil }
func (s *ScalarDatum) Type() arrow.DataType      { return s.Value.DataType() }
func (s *ScalarDatum) String() string            { return fmt.Sprintf("Scalar:{%s}", s.Value) }
func (s *ScalarDatum) NullN() int64 {
	if s.Value.IsValid() {
		return 0
	}
	return 1
}

func (s *ScalarDatum) Descr() ValueDescr {
	return ValueDescr{ShapeScalar, s.Value.DataType()}
}

func (s *ScalarDatum) Equals(other Datum) bool {
	rhs, ok := other.(*ScalarDatum)
	if !ok {
		return false
	}

	return scalar.Equals(s.Value, rhs.Value)
}

type ArrayDatum struct {
	Value *array.Data
}

func (ArrayDatum) Kind() DatumKind               { return KindArray }
func (ArrayDatum) Shape() ValueShape             { return ShapeArray }
func (a *ArrayDatum) Type() arrow.DataType       { return a.Value.DataType() }
func (a *ArrayDatum) Len() int64                 { return int64(a.Value.Len()) }
func (a *ArrayDatum) NullN() int64               { return int64(a.Value.NullN()) }
func (a *ArrayDatum) Descr() ValueDescr          { return ValueDescr{ShapeArray, a.Value.DataType()} }
func (a *ArrayDatum) String() string             { return fmt.Sprintf("Array:{%s}", a.Value.DataType()) }
func (a *ArrayDatum) MakeArray() array.Interface { return array.MakeFromData(a.Value) }
func (c *ArrayDatum) Chunks() []array.Interface  { return []array.Interface{c.MakeArray()} }
func (a *ArrayDatum) Equals(other Datum) bool {
	rhs, ok := other.(*ArrayDatum)
	if !ok {
		return false
	}

	left := a.MakeArray()
	right := rhs.MakeArray()
	defer left.Release()
	defer right.Release()

	return array.ArrayEqual(left, right)
}

type ChunkedDatum struct {
	Value *array.Chunked
}

func (ChunkedDatum) Kind() DatumKind              { return KindChunkedArray }
func (ChunkedDatum) Shape() ValueShape            { return ShapeArray }
func (c *ChunkedDatum) Type() arrow.DataType      { return c.Value.DataType() }
func (c *ChunkedDatum) Len() int64                { return int64(c.Value.Len()) }
func (c *ChunkedDatum) NullN() int64              { return int64(c.Value.NullN()) }
func (c *ChunkedDatum) Descr() ValueDescr         { return ValueDescr{ShapeArray, c.Value.DataType()} }
func (c *ChunkedDatum) String() string            { return fmt.Sprintf("ChunkedArray:{%s}", c.Value.DataType()) }
func (c *ChunkedDatum) Chunks() []array.Interface { return c.Value.Chunks() }
func (c *ChunkedDatum) Equals(other Datum) bool {
	rhs, ok := other.(*ChunkedDatum)
	if !ok {
		return false
	}

	return array.ChunkedEqual(c.Value, rhs.Value)
}

type RecordDatum struct {
	Value array.Record
}

func (RecordDatum) Kind() DatumKind          { return KindRecordBatch }
func (r *RecordDatum) Len() int64            { return r.Value.NumRows() }
func (RecordDatum) String() string           { return "RecordBatch" }
func (r *RecordDatum) Schema() *arrow.Schema { return r.Value.Schema() }
func (r *RecordDatum) Equals(other Datum) bool {
	rhs, ok := other.(*RecordDatum)
	if !ok {
		return false
	}

	return array.RecordEqual(r.Value, rhs.Value)
}

type TableDatum struct {
	Value array.Table
}

func (TableDatum) Kind() DatumKind          { return KindTable }
func (r *TableDatum) Len() int64            { return r.Value.NumRows() }
func (TableDatum) String() string           { return "Table" }
func (r *TableDatum) Schema() *arrow.Schema { return r.Value.Schema() }
func (r *TableDatum) Equals(other Datum) bool {
	rhs, ok := other.(*TableDatum)
	if !ok {
		return false
	}

	return array.TableEqual(r.Value, rhs.Value)
}

type CollectionDatum []Datum

func (CollectionDatum) Kind() DatumKind { return KindCollection }
func (c CollectionDatum) Len() int64    { return int64(len(c)) }
func (c CollectionDatum) String() string {
	var b strings.Builder
	b.WriteString("Collection(")
	for i, d := range c {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(d.String())
	}
	b.WriteByte(')')
	return b.String()
}
func (c CollectionDatum) Equals(other Datum) bool {
	rhs, ok := other.(CollectionDatum)
	if !ok {
		return false
	}

	if len(c) != len(rhs) {
		return false
	}

	for i := range c {
		if !c[i].Equals(rhs[i]) {
			return false
		}
	}
	return true
}

var (
	_ ArrayLikeDatum  = (*ScalarDatum)(nil)
	_ ArrayLikeDatum  = (*ArrayDatum)(nil)
	_ ArrayLikeDatum  = (*ChunkedDatum)(nil)
	_ RecordLikeDatum = (*RecordDatum)(nil)
	_ RecordLikeDatum = (*TableDatum)(nil)
	_ Datum           = (CollectionDatum)(nil)
)
