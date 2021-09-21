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

package tools

import (
	"encoding/json"
	"errors"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
)

func appendToBldr(bldr array.Builder, value interface{}) error {
	switch v := value.(type) {
	case string:
		bldr.(*array.StringBuilder).Append(v)
	case bool:
		bldr.(*array.BooleanBuilder).Append(v)
	case float64:
		switch bldr := bldr.(type) {
		case *array.Int8Builder:
			bldr.Append(int8(v))
		case *array.Uint8Builder:
			bldr.Append(uint8(v))
		case *array.Int16Builder:
			bldr.Append(int16(v))
		case *array.Uint16Builder:
			bldr.Append(uint16(v))
		case *array.Int32Builder:
			bldr.Append(int32(v))
		case *array.Uint32Builder:
			bldr.Append(uint32(v))
		case *array.Int64Builder:
			bldr.Append(int64(v))
		case *array.Uint64Builder:
			bldr.Append(uint64(v))
		case *array.Float32Builder:
			bldr.Append(float32(v))
		case *array.Float64Builder:
			bldr.Append(float64(v))
		default:
			return errors.New("from json type not implemented yet")
		}
	case nil:
		bldr.AppendNull()
	}
	return nil
}

func structFromJSON(typ *arrow.StructType, bldr *array.StructBuilder, values []interface{}) error {
	for _, v := range values {
		if v == nil {
			bldr.AppendNull()
			continue
		}

		bldr.Append(true)
		v := v.(map[string]interface{})
		for i := 0; i < bldr.NumField(); i++ {
			fieldBldr := bldr.FieldBuilder(i)
			field := typ.Field(i)
			if err := appendToBldr(fieldBldr, v[field.Name]); err != nil {
				return err
			}
		}
	}

	return nil
}

func RecordFromJSON(s *arrow.Schema, data []byte) (array.Record, error) {
	var record interface{}
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, err
	}

	raw, ok := record.([]interface{})
	if !ok {
		return nil, errors.New("json record must be an array at the top level")
	}

	typ := arrow.StructOf(s.Fields()...)
	bldr := array.NewStructBuilder(memory.DefaultAllocator, typ)
	defer bldr.Release()

	if err := structFromJSON(typ, bldr, raw); err != nil {
		return nil, err
	}

	arr := bldr.NewStructArray()
	defer arr.Release()

	cols := make([]array.Interface, arr.NumField())
	for i := 0; i < arr.NumField(); i++ {
		cols[i] = arr.Field(i)
	}
	return array.NewRecord(s, cols, -1), nil
}
