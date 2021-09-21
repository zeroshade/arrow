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
	"context"
	"fmt"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/apache/arrow/go/arrow/memory"
)

type RecordMessage struct {
	Record array.Record
	Err    error
}

type RecordGenerator <-chan RecordMessage

type ScanTaskIterator func() (ScanTask, error)

type FragmentMessage struct {
	Fragment Fragment
	Err      error
}

type FragmentIterator <-chan FragmentMessage

type ScanTaskMessage struct {
	Task ScanTask
	Err  error
}

type ScanTaskGenerator <-chan ScanTaskMessage

type FragmentScanOptions interface {
	TypeName() string
	fmt.Stringer
}

type ScanOptions struct {
	Filter, Projection  compute.Expression
	DatasetSchema       *arrow.Schema
	ProjectedSchema     *arrow.Schema
	BatchSize           int64
	BatchReadahead      int32
	FragmentReadahead   int32
	Mem                 memory.Allocator
	Ctx                 context.Context
	UseAsync            bool
	FragmentScanOptions FragmentScanOptions
}

func (s *ScanOptions) MaterializedFields() []string { return nil }

func DefaultScanOptions() *ScanOptions {
	return &ScanOptions{
		BatchSize:         1 << 20,
		BatchReadahead:    32,
		FragmentReadahead: 8,
		Mem:               memory.DefaultAllocator,
		UseAsync:          false,
	}
}

type ScanTask interface {
	Execute() RecordGenerator
	Options() *ScanOptions
	Fragment() Fragment
}

type scanTask struct {
	fragment Fragment
	opts     *ScanOptions
}

func (s *scanTask) Options() *ScanOptions { return s.opts }
func (s *scanTask) Fragment() Fragment    { return s.fragment }

type Fragment interface {
	fmt.Stringer
	ReadPhysicalSchema() (*arrow.Schema, error)
	Scan(opts *ScanOptions) (ScanTaskIterator, error)
	TypeName() string
	PartitionExpr() compute.Expression
}

func makeScanTaskItr(fn func(interface{}) (ScanTask, error), args func() (interface{}, error)) ScanTaskIterator {
	return func() (ScanTask, error) {
		n, err := args()
		if err != nil {
			return nil, err
		}
		return fn(n)
	}
}

func makeRecordGenerator(recs []array.Record) RecordGenerator {
	gen := make(chan RecordMessage)
	go func() {
		defer close(gen)
		for _, r := range recs {
			gen <- RecordMessage{r, nil}
		}
	}()
	return gen
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

type dataset struct {
	schema    *arrow.Schema
	partition compute.Expression
}

func (d *dataset) PartitionExpr() compute.Expression { return d.partition }
func (d *dataset) Schema() *arrow.Schema             { return d.schema }

type Dataset interface {
	Schema() *arrow.Schema
	PartitionExpr() compute.Expression
	TypeName() string
	ReplaceSchema(*arrow.Schema) (Dataset, error)
	GetFragments() (FragmentIterator, error)
	GetFragmentsCond(predicate compute.Expression) (FragmentIterator, error)
}

func GetFragmentsFromDatasets(ds []Dataset, predicate compute.Expression) FragmentIterator {
	fragItr := make(chan FragmentMessage)
	go func() {
		defer close(fragItr)
		for _, d := range ds {
			itr, err := d.GetFragmentsCond(predicate)
			if err != nil {
				fragItr <- FragmentMessage{nil, err}
				break
			}

			for f := range itr {
				fragItr <- f
				if f.Err != nil {
					return
				}
			}
		}
	}()
	return fragItr
}

func SchemaFromColumnNames(input *arrow.Schema, names []string) *arrow.Schema {
	cols := make([]arrow.Field, 0, len(names))
	for _, n := range names {
		ref := compute.NewFieldNameRef(n)
		field, err := ref.GetOne(input)
		if err == nil {
			cols = append(cols, *field)
		}
	}

	meta := input.Metadata()
	return arrow.NewSchema(cols, &meta)
}
