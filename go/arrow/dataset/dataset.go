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
	"io"
	"math"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/apache/arrow/go/arrow/memory"
	"golang.org/x/xerrors"
)

type RecordGenerator <-chan array.Record

type RecordIterator func() (array.Record, error)

type ScanTaskIterator func() (ScanTask, error)

type FragmentIterator func() (Fragment, error)

var emptyFragmentItr = func() (Fragment, error) { return nil, nil }

type FragmentScanOptions interface {
	TypeName() string
	fmt.Stringer
}

type ScanOptions struct {
	Filter, Projection  compute.BoundExpression
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
	Execute() RecordIterator
	Options() *ScanOptions
	Fragment() Fragment
}

type scanTask struct {
	fragment Fragment
	opts     *ScanOptions
}

func (s *scanTask) Options() *ScanOptions { return s.opts }
func (s *scanTask) Fragment() Fragment    { return s.fragment }

type InMemoryScanTask struct {
	scanTask
	batches []array.Record
}

func (i *InMemoryScanTask) Execute() RecordIterator {
	return makeRecordIterator(i.batches)
}

type Fragment interface {
	fmt.Stringer
	ReadPhysicalSchema() (*arrow.Schema, error)
	Scan(opts *ScanOptions) (ScanTaskIterator, error)
	ScanBatchesAsync(opts *ScanOptions) (RecordIterator, error)
	TypeName() string
	PartitionExpr() compute.Expression
}

type InMemoryFragment struct {
	schema    *arrow.Schema
	batches   []array.Record
	partition compute.Expression
}

func NewInMemoryFragment(batches []array.Record) *InMemoryFragment {
	return &InMemoryFragment{schema: batches[0].Schema(), batches: batches, partition: compute.NewLiteral(true)}
}

func (f *InMemoryFragment) ReadPhysicalSchema() (*arrow.Schema, error) {
	return f.schema, nil
}

func (f *InMemoryFragment) ScanBatchesAsync(opt *ScanOptions) (RecordIterator, error) {
	return nil, nil
}

func (f *InMemoryFragment) TypeName() string                  { return "in-memory" }
func (f *InMemoryFragment) PartitionExpr() compute.Expression { return f.partition }
func (f *InMemoryFragment) String() string                    { return "" }

func makeScanTaskItr(fn func(interface{}) (ScanTask, error), args func() (interface{}, error)) ScanTaskIterator {
	return func() (ScanTask, error) {
		n, err := args()
		if err != nil {
			return nil, err
		}
		return fn(n)
	}
}

func makeRecordIterator(recs []array.Record) RecordIterator {
	i := 0
	return func() (array.Record, error) {
		if i >= len(recs) {
			return nil, io.EOF
		}

		ret := recs[i]
		i++
		return ret, nil
	}
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func (in *InMemoryFragment) Scan(opts *ScanOptions) (ScanTaskIterator, error) {
	batchSz := opts.BatchSize
	recItr := makeRecordIterator(in.batches)
	recordsFn := func() (interface{}, error) {
		return recItr()
	}
	return makeScanTaskItr(func(i interface{}) (ScanTask, error) {
		rb := i.(array.Record)
		defer rb.Release()

		nBatches := int(math.Ceil(float64(rb.NumRows()) / float64(batchSz)))

		batches := make([]array.Record, nBatches)
		for i := range batches {
			start := batchSz * int64(i)
			batches[i] = rb.NewSlice(start, min(rb.NumRows(), start+batchSz))
		}

		return &InMemoryScanTask{
			scanTask: scanTask{opts: opts, fragment: in},
			batches:  batches}, nil
	}, recordsFn), nil
}

type datasetImpl interface {
	getFragmentsImpl(predicate compute.Expression) (FragmentIterator, error)

	Schema() *arrow.Schema
	TypeName() string
	ReplaceSchema(schema *arrow.Schema) (*Dataset, error)
}

type Dataset struct {
	datasetImpl

	partition compute.Expression
}

func (d *Dataset) PartitionExpr() compute.Expression { return d.partition }
func (d *Dataset) GetFragments() (FragmentIterator, error) {
	return d.GetFragmentsCond(compute.NewLiteral(true))
}

func (d *Dataset) GetFragmentsCond(predicate compute.Expression) (FragmentIterator, error) {
	pred, err := compute.SimplifyWithGuarantee(predicate, d.partition)
	if err != nil {
		return nil, err
	}

	if pred.IsSatisfiable() {
		return d.datasetImpl.getFragmentsImpl(pred)
	}

	return emptyFragmentItr, nil
}

func NewInMemoryDataset(schema *arrow.Schema, gen func() RecordIterator) *Dataset {
	return &Dataset{
		&inMemoryImpl{schema, gen},
		compute.NewLiteral(true),
	}
}

func NewStaticInMemoryDataset(schema *arrow.Schema, recs []array.Record) *Dataset {
	return NewInMemoryDataset(schema, func() RecordIterator { return makeRecordIterator(recs) })
}

type inMemoryImpl struct {
	schema  *arrow.Schema
	batches func() RecordIterator
}

func (i *inMemoryImpl) Schema() *arrow.Schema { return i.schema }
func (i *inMemoryImpl) TypeName() string      { return "in-memory" }

func (i *inMemoryImpl) ReplaceSchema(schema *arrow.Schema) (*Dataset, error) {
	if err := checkProjectable(i.schema, schema); err != nil {
		return nil, err
	}

	return &Dataset{&inMemoryImpl{schema, i.batches}, compute.NewLiteral(true)}, nil
}

func (i *inMemoryImpl) getFragmentsImpl(compute.Expression) (FragmentIterator, error) {
	itr := i.batches()
	return func() (Fragment, error) {
		next, err := itr()
		if err != nil {
			return nil, err
		}

		if !next.Schema().Equal(i.schema) {
			return nil, xerrors.Errorf("yielded batch had schema %s which did not match InMemory Source's: %s", next.Schema(), i.schema)
		}

		return NewInMemoryFragment([]array.Record{next}), nil
	}, nil
}
