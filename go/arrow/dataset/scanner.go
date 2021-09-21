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
	"io"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/apache/arrow/go/arrow/internal/debug"
	"github.com/apache/arrow/go/arrow/scalar"
)

type TaggedRecord struct {
	RecordBatch array.Record
	Fragment    Fragment
	Err         error
}

type filterAndProjectScanTask struct {
	ScanTask
	partition compute.Expression
}

func (ft *filterAndProjectScanTask) Execute() RecordGenerator {

	it := ft.ScanTask.Execute()
	simplifiedFilter, err := compute.SimplifyWithGuarantee(ft.Options().Filter, ft.partition)
	if err != nil {
		out := make(chan RecordMessage)
		out <- RecordMessage{Err: err}
		close(out)
		return out
	}

	simplifiedProjection, err := compute.SimplifyWithGuarantee(ft.Options().Projection, ft.partition)
	if err != nil {
		out := make(chan RecordMessage)
		out <- RecordMessage{Err: err}
		close(out)
		return out
	}

	filterGen := filterRecordBatches(it, simplifiedFilter, ft.Options())
	return projectRecordBatches(filterGen, simplifiedProjection, ft.Options())
}

func getScanTaskGenerator(frag FragmentIterator, opts *ScanOptions) ScanTaskGenerator {
	gen := make(chan ScanTaskMessage, opts.FragmentReadahead)
	go func() {
		defer close(gen)

		for f := range frag {
			if f.Err != nil {
				gen <- ScanTaskMessage{nil, f.Err}
				break
			}

			scanItr, err := f.Fragment.Scan(opts)
			if err != nil {
				gen <- ScanTaskMessage{nil, err}
				break
			}

			partition := f.Fragment.PartitionExpr()
			for {
				st, err := scanItr()
				if err != nil {
					if err != io.EOF {
						gen <- ScanTaskMessage{nil, err}
					}
					break
				}

				gen <- ScanTaskMessage{&filterAndProjectScanTask{st, partition}, nil}
			}
		}
	}()
	return gen
}

type Scanner struct {
	options *ScanOptions
	dataset Dataset
}

func NewScanner(opts *ScanOptions, dataset Dataset) (*Scanner, error) {
	opts.DatasetSchema = dataset.Schema()
	if !opts.Filter.IsBound() {
		if err := SetFilter(opts, compute.NewLiteral(true)); err != nil {
			return nil, err
		}
	}
	if !opts.Projection.IsBound() {
		if err := SetProjectionSchema(opts, dataset.Schema()); err != nil {
			return nil, err
		}
	}

	return &Scanner{opts, dataset}, nil
}

func (s *Scanner) getFragments() FragmentIterator {
	return GetFragmentsFromDatasets([]Dataset{s.dataset}, s.options.Filter)
}

func (s *Scanner) Scan() ScanTaskGenerator {
	fragItr := s.getFragments()
	return getScanTaskGenerator(fragItr, s.options)
}

func (s *Scanner) ScanBatches() <-chan TaggedRecord {
	gen := s.Scan()

	out := make(chan TaggedRecord, s.options.BatchReadahead)
	go func() {
		defer close(out)
		// var wg sync.WaitGroup
		// defer wg.Wait()

		for st := range gen {
			if st.Err != nil {
				out <- TaggedRecord{nil, nil, st.Err}
				break
			}

			// wg.Add(1)
			// go func(task ScanTask) {
			// defer wg.Done()

			for m := range st.Task.Execute() {
				if m.Err != nil {
					out <- TaggedRecord{nil, st.Task.Fragment(), m.Err}
					break
				}
				out <- TaggedRecord{m.Record, st.Task.Fragment(), nil}
			}
			// }(st.Task)
		}
	}()
	return out
}

func filterRecord(in array.Record, filter compute.Expression, opts *ScanOptions) (array.Record, error) {
	mask, err := compute.ExecuteScalarExprWithSchema(opts.Ctx, opts.Mem, opts.DatasetSchema, in, filter)
	if err != nil {
		return nil, err
	}

	if ms, ok := mask.(*compute.ScalarDatum); ok {
		if ms.Value.IsValid() && ms.Value.(*scalar.Boolean).Value {
			return in, nil
		}
		defer in.Release()
		return in.NewSlice(0, 0), nil
	}

	defer in.Release()

	input := compute.NewDatum(in)
	defer input.Release()

	filtered, err := compute.Filter(opts.Ctx, opts.Mem, input, mask, compute.FilterOptions{})
	if err != nil {
		return nil, err
	}

	return filtered.(*compute.RecordDatum).Value, err
}

func filterRecordBatches(it RecordGenerator, filter compute.Expression, opts *ScanOptions) RecordGenerator {
	out := make(chan RecordMessage)
	go func() {
		defer close(out)

		for msg := range it {
			if msg.Err != nil {
				out <- msg
				break
			}

			msg.Record, msg.Err = filterRecord(msg.Record, filter, opts)
			out <- msg
			if msg.Err != nil {
				break
			}
		}
	}()
	return out
}

func projectRecord(in array.Record, projection compute.Expression, opts *ScanOptions) (array.Record, error) {
	projected, err := compute.ExecuteScalarExprWithSchema(opts.Ctx, opts.Mem, opts.DatasetSchema, in, projection)
	if err != nil {
		return nil, err
	}
	defer projected.Release()

	var arr array.Interface

	debug.Assert(projected.(compute.ArrayLikeDatum).Type().ID() == arrow.STRUCT, "invalid return from projection")
	if projected.(compute.ArrayLikeDatum).Descr().Shape == compute.ShapeScalar {
		arr, err = scalar.MakeArrayFromScalar(projected.(*compute.ScalarDatum).Value, int(in.NumRows()), opts.Mem)
		if err != nil {
			return nil, err
		}
	} else {
		arr = projected.(*compute.ArrayDatum).MakeArray()
	}
	defer arr.Release()

	out := array.RecordFromStructArray(arr.(*array.Struct))
	defer out.Release()

	meta := in.Schema().Metadata()
	newSchema := arrow.NewSchema(out.Schema().Fields(), &meta)
	return array.NewRecord(newSchema, out.Columns(), out.NumRows()), nil
}

func projectRecordBatches(it RecordGenerator, projection compute.Expression, opts *ScanOptions) RecordGenerator {
	out := make(chan RecordMessage)
	go func() {
		defer close(out)

		for msg := range it {
			if msg.Err != nil {
				out <- msg
				break
			}

			msg.Record, msg.Err = projectRecord(msg.Record, projection, opts)
			out <- msg
			if msg.Err != nil {
				break
			}
		}
	}()
	return out
}
