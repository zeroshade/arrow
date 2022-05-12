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

package kernels

import (
	"errors"
	"fmt"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/array"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
)

var (
	intTypes = []arrow.DataType{
		arrow.PrimitiveTypes.Int8,
		arrow.PrimitiveTypes.Uint8,
		arrow.PrimitiveTypes.Int16,
		arrow.PrimitiveTypes.Uint16,
		arrow.PrimitiveTypes.Int32,
		arrow.PrimitiveTypes.Uint32,
		arrow.PrimitiveTypes.Int64,
		arrow.PrimitiveTypes.Uint64,
	}
	floatingTypes = []arrow.DataType{
		arrow.PrimitiveTypes.Float32,
		arrow.PrimitiveTypes.Float64,
	}
	temporalTypes = []arrow.DataType{
		arrow.FixedWidthTypes.Date32,
		arrow.FixedWidthTypes.Date64,
		arrow.FixedWidthTypes.Time32s,
		arrow.FixedWidthTypes.Time32ms,
		arrow.FixedWidthTypes.Time64us,
		arrow.FixedWidthTypes.Time64ns,
		arrow.FixedWidthTypes.Timestamp_s,
		arrow.FixedWidthTypes.Timestamp_ms,
		arrow.FixedWidthTypes.Timestamp_us,
		arrow.FixedWidthTypes.Timestamp_ns,
	}
	intervalTypes = []arrow.DataType{
		arrow.FixedWidthTypes.DayTimeInterval,
		arrow.FixedWidthTypes.MonthInterval,
		arrow.FixedWidthTypes.MonthDayNanoInterval,
	}
	baseBinaryTypes = []arrow.DataType{
		arrow.BinaryTypes.Binary,
		arrow.BinaryTypes.String,
	}
	numericTypes = append(append(intTypes, floatingTypes...), arrow.FixedWidthTypes.Boolean)
)

func canCastFromDictionary(id arrow.Type) bool {
	return arrow.IsPrimitive(id) || arrow.IsBinaryLike(id) || arrow.IsFixedSizeBinary(id)
}

func unpackDictionary(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
	debug.Assert(out.Kind() == compute.KindArray, "invalid unpack dictionary type")

	dictArr := batch.Values[0].(*compute.ArrayDatum).MakeArray().(*array.Dictionary)
	defer dictArr.Release()

	opts := ctx.State.(*compute.CastOptions)
	dictType := dictArr.Dictionary().DataType()
	if !arrow.TypeEqual(dictType, opts.ToType) && !CanCast(dictType, opts.ToType) {
		return fmt.Errorf("cast type %s incompatible with dictionary type %s", opts.ToType, dictType)
	}

	return errors.New("casting dictionaries not yet implemented")
}
