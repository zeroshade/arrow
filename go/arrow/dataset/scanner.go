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
	"sync"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
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
	dataset *Dataset
}

func NewScanner(opts *ScanOptions, dataset *Dataset) (*Scanner, error) {
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
	return GetFragmentsFromDatasets([]*Dataset{s.dataset}, s.options.Filter)
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
		var wg sync.WaitGroup
		defer wg.Wait()

		for st := range gen {
			if st.Err != nil {
				out <- TaggedRecord{nil, nil, st.Err}
				break
			}

			wg.Add(1)
			go func(task ScanTask) {
				defer wg.Done()

				for m := range task.Execute() {
					if m.Err != nil {
						out <- TaggedRecord{nil, task.Fragment(), m.Err}
						break
					}
					out <- TaggedRecord{m.Record, task.Fragment(), nil}
				}
			}(st.Task)
		}
	}()
	return out
}
