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
	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/internal"
	"github.com/apache/arrow/go/v9/arrow/internal/debug"
)

const (
	millisecondsInDay int64 = 86400000
)

func getDate32Cast() CastFunction {
	fn := NewCastFunction("cast_date32", arrow.DATE32)
	outType := arrow.PrimitiveTypes.Date32
	outputType := functions.NewOutputType(outType)
	addCommonCasts(arrow.DATE32, outputType, &fn)
	// int32 -> date32
	addZeroCopyCast(arrow.INT32, functions.NewExactInput(arrow.PrimitiveTypes.Int32, compute.ShapeAny), outputType, &fn)

	fn.AddNewKernel(arrow.DATE64, []functions.InputType{functions.NewExactInput(arrow.PrimitiveTypes.Date64, compute.ShapeAny)}, outputType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "date64 cast requires array datum")
			return internal.ShiftTime[int64, int32](ctx, arrow.ConvDIVIDE, millisecondsInDay, batch.Values[0].(*compute.ArrayDatum).Value, out.(*compute.ArrayDatum).Value)
		}, functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)

	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outputType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "timestamp cast requires array datum")
			return internal.TimestampToDate32(ctx, batch, out)
		}, functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)
	return fn
}

func getDate64Cast() CastFunction {
	fn := NewCastFunction("cast_date64", arrow.DATE64)
	outType := arrow.PrimitiveTypes.Date64
	outputType := functions.NewOutputType(outType)
	addCommonCasts(arrow.DATE64, outputType, &fn)
	// int64 -> date64
	addZeroCopyCast(arrow.INT64, functions.NewExactInput(arrow.PrimitiveTypes.Int64, compute.ShapeAny), outputType, &fn)

	fn.AddNewKernel(arrow.DATE32, []functions.InputType{functions.NewExactInput(arrow.FixedWidthTypes.Date32, compute.ShapeAny)}, outputType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "date32 -> date64 cast must use an array")
			return internal.ShiftTime[int32, int64](ctx, arrow.ConvMULTIPLY, millisecondsInDay, batch.Values[0].(*compute.ArrayDatum).Value, out.(*compute.ArrayDatum).Value)
		}, functions.NullIntersection), functions.NullIntersection, functions.MemPrealloc)

	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outputType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "timestamp cast requires array datum")
			return internal.TimestampToDate64(ctx, batch, out)
		}, functions.NullIntersection), functions.NullIntersection, functions.MemPrealloc)
	return fn
}

func getDurationCast() CastFunction {
	fn := NewCastFunction("cast_duration", arrow.DURATION)
	addCommonCasts(arrow.DURATION, outputTargetType, &fn)
	// same integer representation
	addZeroCopyCast(arrow.INT64, functions.NewExactInput(arrow.PrimitiveTypes.Int64, compute.ShapeAny), outputTargetType, &fn)

	// between durations
	kernel := functions.NewScalarKernelWithSig(
		functions.NewKernelSig([]functions.InputType{functions.NewInputIDType(arrow.DURATION)}, outputTargetType, false),
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Duration, arrow.Duration], functions.NullIntersection), nil)
	fn.AddKernel(arrow.DURATION, kernel)
	return fn
}

func getIntervalCast() CastFunction {
	fn := NewCastFunction("cast_month_day_nano_interval", arrow.INTERVAL_MONTH_DAY_NANO)
	addCommonCasts(arrow.INTERVAL_MONTH_DAY_NANO, outputTargetType, &fn)
	return fn
}

func getTime32Cast() CastFunction {
	fn := NewCastFunction("cast_time32", arrow.TIME32)
	addCommonCasts(arrow.TIME32, outputTargetType, &fn)

	// zero copy when the unit is the same or same integer representation
	addZeroCopyCast(arrow.INT32, functions.NewExactInput(arrow.PrimitiveTypes.Int32, compute.ShapeAny), outputTargetType, &fn)
	fn.AddNewKernel(arrow.TIME32, []functions.InputType{functions.NewInputIDType(arrow.TIME64)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Time64, arrow.Time32], functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)

	kernel := functions.NewScalarKernelWithSig(
		functions.NewKernelSig([]functions.InputType{functions.NewInputIDType(arrow.TIME32)}, outputTargetType, false),
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Time32, arrow.Time32], functions.NullIntersection), nil)
	fn.AddKernel(arrow.TIME32, kernel)

	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(internal.TimestampToTime32, functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)
	return fn
}

func getTime64Cast() CastFunction {
	fn := NewCastFunction("cast_time64", arrow.TIME64)
	addCommonCasts(arrow.TIME64, outputTargetType, &fn)

	// zero copy when unit is the same or same integer representation
	addZeroCopyCast(arrow.INT64, functions.NewExactInput(arrow.PrimitiveTypes.Int64, compute.ShapeAny), outputTargetType, &fn)

	// time32 -> time64
	fn.AddNewKernel(arrow.TIME64, []functions.InputType{functions.NewInputIDType(arrow.TIME32)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Time32, arrow.Time64], functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)

	kernel := functions.NewScalarKernelWithSig(
		functions.NewKernelSig([]functions.InputType{functions.NewInputIDType(arrow.TIME64)}, outputTargetType, false),
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Time64, arrow.Time64], functions.NullIntersection), nil)
	fn.AddKernel(arrow.TIME64, kernel)

	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(internal.TimestampToTime64, functions.NullIntersection),
		functions.NullIntersection, functions.MemPrealloc)
	return fn
}

func getTimestampCast() CastFunction {
	fn := NewCastFunction("cast_timestamp", arrow.TIMESTAMP)
	addCommonCasts(arrow.TIMESTAMP, outputTargetType, &fn)

	// same integer representation
	addZeroCopyCast(arrow.INT64, functions.NewExactInput(arrow.PrimitiveTypes.Int64, compute.ShapeAny), outputTargetType, &fn)

	// from date types
	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.DATE32)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "should be array datum in timestamp cast")

			outType := out.Type().(*arrow.TimestampType)
			_, factor := arrow.GetTimestampConvert(arrow.Second, outType.Unit)
			// multiply to achieve days -> unit
			factor *= millisecondsInDay / 1000

			return internal.ShiftTime[int32, int64](ctx, arrow.ConvMULTIPLY, factor, batch.Values[0].(*compute.ArrayDatum).Value, out.(*compute.ArrayDatum).Value)
		}, functions.NullIntersection), functions.NullIntersection, functions.MemPrealloc)

	fn.AddNewKernel(arrow.TIMESTAMP, []functions.InputType{functions.NewInputIDType(arrow.DATE64)}, outputTargetType,
		trivialScalarUnaryAsArrayExec(func(ctx *functions.KernelCtx, batch *functions.ExecBatch, out compute.Datum) error {
			debug.Assert(batch.Values[0].Kind() == compute.KindArray, "should be array datum in timestamp cast")

			outType := out.Type().(*arrow.TimestampType)
			op, factor := arrow.GetTimestampConvert(arrow.Millisecond, outType.Unit)

			return internal.ShiftTime[int32, int64](ctx, op, factor, batch.Values[0].(*compute.ArrayDatum).Value, out.(*compute.ArrayDatum).Value)
		}, functions.NullIntersection), functions.NullIntersection, functions.MemPrealloc)

	kernel := functions.NewScalarKernelWithSig(
		functions.NewKernelSig([]functions.InputType{functions.NewInputIDType(arrow.TIMESTAMP)}, outputTargetType, false),
		trivialScalarUnaryAsArrayExec(internal.SimpleTemporalCast[arrow.Timestamp, arrow.Timestamp], functions.NullIntersection), nil)
	fn.AddKernel(arrow.TIMESTAMP, kernel)
	return fn
}

func getTemporalCasts() (out []CastFunction) {
	out = make([]CastFunction, 0)
	out = append(out, getDate32Cast())
	out = append(out, getDate64Cast())
	out = append(out, getTime32Cast())
	out = append(out, getTime64Cast())
	out = append(out, getTimestampCast())
	out = append(out, getDurationCast())
	return out
}
