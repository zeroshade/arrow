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
	"math"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/pkg/errors"
)

type InMemoryScanTask struct {
	scanTask
	batches []array.Record
}

func (i *InMemoryScanTask) Execute() RecordGenerator {
	return makeRecordGenerator(i.batches)
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

func (f *InMemoryFragment) TypeName() string                  { return "in-memory" }
func (f *InMemoryFragment) PartitionExpr() compute.Expression { return f.partition }
func (f *InMemoryFragment) String() string                    { return "" }

func (in *InMemoryFragment) Scan(opts *ScanOptions) (ScanTaskIterator, error) {
	batchSz := opts.BatchSize
	recgen := makeRecordGenerator(in.batches)
	recordsFn := func() (interface{}, error) {
		rec, ok := <-recgen
		if !ok {
			return nil, io.EOF
		}
		return rec.Record, rec.Err
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

func NewInMemoryDataset(schema *arrow.Schema, recs []array.Record) *Dataset {
	return &Dataset{&inMemoryImpl{schema, recs}, compute.NewLiteral(true)}
}

type inMemoryImpl struct {
	schema  *arrow.Schema
	batches []array.Record
}

func (i *inMemoryImpl) Schema() *arrow.Schema { return i.schema }
func (i *inMemoryImpl) TypeName() string      { return "in-memory" }

func (i *inMemoryImpl) ReplaceSchema(schema *arrow.Schema) (*Dataset, error) {
	if err := checkProjectable(i.schema, schema); err != nil {
		return nil, err
	}

	return &Dataset{&inMemoryImpl{schema, i.batches}, compute.NewLiteral(true)}, nil
}

func (i *inMemoryImpl) getFragmentsImpl(compute.BoundExpression) FragmentIterator {
	itr := make(chan FragmentMessage)
	go func() {
		defer close(itr)
		for _, rec := range i.batches {
			if !rec.Schema().Equal(i.schema) {
				itr <- FragmentMessage{nil, errors.Errorf("yielded batch had schema %s which did not match InMemory Source's: %s", rec.Schema(), i.schema)}
				break
			}

			itr <- FragmentMessage{NewInMemoryFragment([]array.Record{rec}), nil}
		}
	}()
	return itr
}
