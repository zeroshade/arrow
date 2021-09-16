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

package dataset

import (
	"testing"

	"github.com/apache/arrow/go/arrow"
	"github.com/stretchr/testify/assert"
)

func TestProjectable(t *testing.T) {
	var projectableTo assert.ComparisonAssertionFunc = func(tt assert.TestingT, from, to interface{}, substr ...interface{}) bool {
		schemaFrom := arrow.NewSchema(from.([]arrow.Field), nil)
		schemaTo := arrow.NewSchema(to.([]arrow.Field), nil)

		return assert.NoError(t, checkProjectable(schemaFrom, schemaTo))
	}

	var notProjectableTo assert.ComparisonAssertionFunc = func(tt assert.TestingT, from, to interface{}, substr ...interface{}) bool {
		schemaFrom := arrow.NewSchema(from.([]arrow.Field), nil)
		schemaTo := arrow.NewSchema(to.([]arrow.Field), nil)

		err := checkProjectable(schemaFrom, schemaTo)
		return assert.ErrorIs(t, err, TypeError) && assert.Contains(t, err.Error(), substr[0])
	}

	i8 := arrow.Field{Name: "i8", Type: arrow.PrimitiveTypes.Int8, Nullable: true}
	u16 := arrow.Field{Name: "u16", Type: arrow.PrimitiveTypes.Uint16, Nullable: true}
	str := arrow.Field{Name: "str", Type: arrow.BinaryTypes.String, Nullable: true}
	i8Req := arrow.Field{Name: "i8", Type: arrow.PrimitiveTypes.Int8}
	u16Req := arrow.Field{Name: "u16", Type: arrow.PrimitiveTypes.Uint16}
	strReq := arrow.Field{Name: "str", Type: arrow.BinaryTypes.String}
	strNil := arrow.Field{Name: "str", Type: arrow.Null}

	tests := []struct {
		name       string
		fieldsFrom []arrow.Field
		fieldsTo   []arrow.Field
		substr     string
		comp       assert.ComparisonAssertionFunc
	}{
		{"trivial", []arrow.Field{}, []arrow.Field{}, "", projectableTo},
		{"trivial single", []arrow.Field{i8}, []arrow.Field{i8}, "", projectableTo},
		{"trivial two", []arrow.Field{i8, u16Req}, []arrow.Field{u16Req}, "", projectableTo},
		{"reorder two", []arrow.Field{i8, u16}, []arrow.Field{u16, i8}, "", projectableTo},
		{"reorder three", []arrow.Field{i8, str, u16}, []arrow.Field{u16, i8, str}, "", projectableTo},
		{"drop field", []arrow.Field{i8}, []arrow.Field{}, "", projectableTo},
		{"add field", []arrow.Field{}, []arrow.Field{i8}, "", projectableTo},
		{"add two fields", []arrow.Field{}, []arrow.Field{i8, u16}, "", projectableTo},
		{"cannot add non nullable", []arrow.Field{}, []arrow.Field{u16Req}, "is not nullable and does not exist in origin schema", notProjectableTo},
		{"cannot add reorder nullable", []arrow.Field{i8}, []arrow.Field{u16Req, i8}, "", notProjectableTo},
		{"cannot change to required", []arrow.Field{i8}, []arrow.Field{i8Req}, "not nullable but is not required in origin schema", notProjectableTo},
		{"can make nullable", []arrow.Field{i8Req}, []arrow.Field{i8}, "", projectableTo},
		{"can convert from null", []arrow.Field{strNil}, []arrow.Field{str}, "", projectableTo},
		{"cannot convert from null to non-nullable", []arrow.Field{strNil}, []arrow.Field{strReq}, "", notProjectableTo},
		{"cannot change type", []arrow.Field{i8}, []arrow.Field{{Name: "i8", Type: arrow.BinaryTypes.String, Nullable: true}}, "fields had matching names but differing types", notProjectableTo},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.comp(t, tt.fieldsFrom, tt.fieldsTo, tt.substr)
		})
	}
}
