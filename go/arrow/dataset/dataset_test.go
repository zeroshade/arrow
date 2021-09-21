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

package dataset_test

import (
	"context"
	"io"
	"io/ioutil"
	"reflect"
	"testing"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/apache/arrow/go/arrow/dataset"
	"github.com/apache/arrow/go/arrow/internal/testing/tools"
	"github.com/apache/arrow/go/arrow/internal/testing/types"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/stretchr/testify/suite"
)

var typmap = map[arrow.Type]reflect.Type{
	arrow.BOOL:    reflect.TypeOf(true),
	arrow.INT32:   reflect.TypeOf(int32(0)),
	arrow.FLOAT64: reflect.TypeOf(float64(0)),
}

func constantArrayGen(b array.Builder, v interface{}, sz int) {
	switch b := b.(type) {
	case *array.BooleanBuilder:
		b.AppendValues(v.([]bool), nil)
	case *array.Int32Builder:
		b.AppendValues(v.([]int32), nil)
	case *array.Float64Builder:
		b.AppendValues(v.([]float64), nil)
	}
}

func constantZeroArrayGen(b array.Builder, dt arrow.DataType, sz int) {
	constantArrayGen(b, reflect.MakeSlice(reflect.SliceOf(typmap[dt.ID()]), sz, sz).Interface(), sz)
}

func ConstantRecordBatchGenZeros(sz int, schema *arrow.Schema) array.Record {
	bldr := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	defer bldr.Release()

	bldr.Reserve(sz)
	for i, b := range bldr.Fields() {
		constantZeroArrayGen(b, schema.Field(i).Type, sz)
	}

	return bldr.NewRecord()
}

func ConstantRecordBatchGenRepeat(rb array.Record, times int) array.RecordReader {
	rdr, _ := array.NewRecordReader(rb.Schema(), createBatchList(rb, times))
	return rdr
}

type DatasetTestSuite struct {
	suite.Suite

	ctx context.Context

	sc   *arrow.Schema
	opts dataset.ScanOptions
}

func TestDatasets(t *testing.T) {
	suite.Run(t, new(InMemoryFragmentSuite))
	suite.Run(t, new(InMemoryDatasetSuite))
	suite.Run(t, new(UnionDatasetSuite))
	suite.Run(t, new(EndToEndSuite))
}

type InMemoryFragmentSuite struct {
	DatasetTestSuite
}

func (d *DatasetTestSuite) SetupTest()    { d.ctx = compute.ExecContext(context.Background()) }
func (d *DatasetTestSuite) TearDownTest() { compute.ReleaseContext(d.ctx) }

func (d *DatasetTestSuite) setFilter(filter compute.Expression) {
	d.opts.Filter = compute.BindExpression(d.ctx, d.opts.Mem, filter, d.sc)
}

func (d *DatasetTestSuite) setSchema(sc *arrow.Schema) {
	d.sc = sc
	d.opts = *dataset.DefaultScanOptions()
	d.opts.Ctx = d.ctx
	d.opts.DatasetSchema = sc
	names := make([]string, len(d.sc.Fields()))
	for i := range names {
		names[i] = d.sc.Field(i).Name
	}
	d.NoError(dataset.SetProjectionNames(&d.opts, names))
	d.setFilter(compute.NewLiteral(true))
}

func (d *DatasetTestSuite) assertScanTaskEquals(expected array.RecordReader, task dataset.ScanTask, ensureDrained bool) {
	for rm := range task.Execute() {
		d.NotNil(rm.Record)
		d.NoError(rm.Err)
		rb := rm.Record

		d.True(expected.Next())
		lhs := expected.Record()
		d.NotNil(lhs)

		d.True(array.RecordEqual(rb, lhs))
		lhs.Release()
		rb.Release()
	}

	if ensureDrained {
		d.False(expected.Next())
	}
}

func (d *DatasetTestSuite) assertFragmentEquals(rdr array.RecordReader, f *dataset.InMemoryFragment, ensureDrained bool) {
	it, err := f.Scan(&d.opts)
	d.NoError(err)

	for {
		task, err := it()
		if err != nil {
			d.ErrorIs(err, io.EOF)
			break
		}
		d.NotNil(task)
		d.assertScanTaskEquals(rdr, task, false)
	}

	if ensureDrained {
		d.False(rdr.Next())
	}
}

func (d *DatasetTestSuite) assertScannerEquals(rdr array.RecordReader, scanner *dataset.Scanner, ensureDrained bool) {
	ch := scanner.ScanBatches()
	for rec := range ch {
		d.True(rdr.Next())
		lhs := rdr.Record()
		d.NotNil(lhs)

		d.Truef(array.RecordEqual(lhs, rec.RecordBatch), "expected: %s\ngot: %s", lhs, rec.RecordBatch)
		lhs.Release()
		rec.RecordBatch.Release()
	}

	if ensureDrained {
		d.False(rdr.Next())
	}
}

func (d *DatasetTestSuite) assertDatasetEquals(rdr array.RecordReader, ds dataset.Dataset, ensureDrained bool) {
	scanner, err := dataset.NewScanner(&d.opts, ds)
	d.NoError(err)

	d.assertScannerEquals(rdr, scanner, true)
	if ensureDrained {
		d.False(rdr.Next())
	}
}

func createBatchList(rb array.Record, nbatches int) []array.Record {
	batchlist := make([]array.Record, nbatches)
	for i := range batchlist {
		batchlist[i] = rb
		rb.Retain()
	}
	return batchlist
}

func (d *InMemoryFragmentSuite) TestScan() {
	const (
		batchSize = 1024
		nbatches  = 16
	)

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil)
	d.setSchema(schema)

	batch := ConstantRecordBatchGenZeros(batchSize, schema)
	defer batch.Release()

	rdr := ConstantRecordBatchGenRepeat(batch, nbatches)
	defer rdr.Release()

	fragment := dataset.NewInMemoryFragment(createBatchList(batch, nbatches))
	d.assertFragmentEquals(rdr, fragment, true)
}

type InMemoryDatasetSuite struct {
	DatasetTestSuite
}

func (d *InMemoryDatasetSuite) TestReplaceSchema() {
	const (
		batchSize = 1
		nbatches  = 1
	)

	d.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil))
	batch := ConstantRecordBatchGenZeros(batchSize, d.sc)
	defer batch.Release()
	rdr := ConstantRecordBatchGenRepeat(batch, nbatches)
	defer rdr.Release()

	ds := dataset.NewInMemoryDataset(d.sc, createBatchList(batch, nbatches))

	// drop field
	_, err := ds.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true}}, nil))
	d.NoError(err)

	// add field (will be materialized as null during projection)
	_, err = ds.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "str", Type: arrow.BinaryTypes.String, Nullable: true}}, nil))
	d.NoError(err)

	// incompatible type
	_, err = ds.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "i32", Type: arrow.BinaryTypes.String}}, nil))
	d.ErrorIs(err, dataset.TypeError)

	// incompatible nullability
	_, err = ds.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: false}}, nil))
	d.ErrorIs(err, dataset.TypeError)

	// add non-nullable field
	_, err = ds.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "str", Type: arrow.BinaryTypes.String}}, nil))
	d.ErrorIs(err, dataset.TypeError)
}

func (d *InMemoryDatasetSuite) TestGetFragments() {
	const (
		batchSize = 1024
		nbatches  = 16
	)

	d.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil))
	batch := ConstantRecordBatchGenZeros(batchSize, d.sc)
	defer batch.Release()
	rdr := ConstantRecordBatchGenRepeat(batch, nbatches)
	defer rdr.Release()

	ds := dataset.NewInMemoryDataset(d.sc, createBatchList(batch, nbatches))
	d.assertDatasetEquals(rdr, ds, true)
}

type UnionDatasetSuite struct {
	DatasetTestSuite
}

func (u *UnionDatasetSuite) TestReplaceSchema() {
	const (
		batchSize = 1
		nbatches  = 1
	)

	u.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil))
	batch := ConstantRecordBatchGenZeros(batchSize, u.sc)
	defer batch.Release()

	children := []dataset.Dataset{
		dataset.NewInMemoryDataset(u.sc, createBatchList(batch, nbatches)),
		dataset.NewInMemoryDataset(u.sc, createBatchList(batch, nbatches)),
	}
	totalBatches := len(children) * nbatches
	rdr := ConstantRecordBatchGenRepeat(batch, totalBatches)
	defer rdr.Release()

	union, err := dataset.NewUnionDataset(u.sc, children)
	u.NoError(err)

	u.assertDatasetEquals(rdr, union, true)

	// drop field
	_, err = union.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true}}, nil))
	u.NoError(err)

	// add field (will be materialized as null during projection)
	_, err = union.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "str", Type: arrow.BinaryTypes.String, Nullable: true}}, nil))
	u.NoError(err)

	// incompatible type
	_, err = union.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "i32", Type: arrow.BinaryTypes.String}}, nil))
	u.ErrorIs(err, dataset.TypeError)

	// incompatible nullability
	_, err = union.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: false}}, nil))
	u.ErrorIs(err, dataset.TypeError)

	// add non-nullable field
	_, err = union.ReplaceSchema(arrow.NewSchema([]arrow.Field{{Name: "str", Type: arrow.BinaryTypes.String}}, nil))
	u.ErrorIs(err, dataset.TypeError)
}

func createDatasetList(ds dataset.Dataset, times int) []dataset.Dataset {
	children := make([]dataset.Dataset, times)
	for i := range children {
		children[i] = ds
	}
	return children
}

func (u *UnionDatasetSuite) TestGetFragments() {
	const (
		batchSize               = 1024
		childPerNode            = 2
		completeBinaryTreeDepth = 4
	)

	u.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil))
	batch := ConstantRecordBatchGenZeros(batchSize, u.sc)
	defer batch.Release()

	const nleaves = uint(1) << completeBinaryTreeDepth
	rdr := ConstantRecordBatchGenRepeat(batch, int(nleaves))
	defer rdr.Release()

	// creates a complete binary tree of depth completeBinaryTreeDepth where
	// the leaves are InMemoryDataset containing childPerNode Fragments
	l1LeafDataset := dataset.NewInMemoryDataset(u.sc, createBatchList(batch, childPerNode))
	l2LeafTreeDataset, err := dataset.NewUnionDataset(u.sc, createDatasetList(l1LeafDataset, childPerNode))
	u.NoError(err)
	l3MiddleTreeDataset, err := dataset.NewUnionDataset(u.sc, createDatasetList(l2LeafTreeDataset, childPerNode))
	u.NoError(err)
	rootDataset, err := dataset.NewUnionDataset(u.sc, createDatasetList(l3MiddleTreeDataset, childPerNode))
	u.NoError(err)

	for i := childPerNode; i < int(nleaves); i++ {
		batch.Retain()
	}

	u.assertDatasetEquals(rdr, rootDataset, true)
}

func (u *UnionDatasetSuite) TestTrivialScan() {
	const (
		nbatches  = 16
		batchSize = 1024
	)

	u.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "i32", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil))
	batch := ConstantRecordBatchGenZeros(batchSize, u.sc)
	defer batch.Release()

	children := []dataset.Dataset{
		dataset.NewInMemoryDataset(u.sc, createBatchList(batch, nbatches)),
		dataset.NewInMemoryDataset(u.sc, createBatchList(batch, nbatches)),
	}

	totalBatches := len(children) * nbatches
	rdr := ConstantRecordBatchGenRepeat(batch, totalBatches)
	defer rdr.Release()

	ds, err := dataset.NewUnionDataset(u.sc, children)
	u.NoError(err)
	u.assertDatasetEquals(rdr, ds, true)
}

type EndToEndSuite struct {
	DatasetTestSuite

	mockfs *tools.MockFS

	expected []string
}

func expectedOutput(schema *arrow.Schema, data []string) []array.Record {
	out := make([]array.Record, len(data))
	for i, d := range data {
		var err error
		out[i], err = types.RecordFromJSON(schema, []byte(d))
		if err != nil {
			panic(err)
		}
	}
	return out
}

func (ete *EndToEndSuite) SetupTest() {
	ete.DatasetTestSuite.SetupTest()
	ete.setSchema(arrow.NewSchema([]arrow.Field{
		{Name: "region", Type: arrow.BinaryTypes.String},
		{Name: "model", Type: arrow.BinaryTypes.String},
		{Name: "sales", Type: arrow.PrimitiveTypes.Float64},
		// partition cols
		{Name: "year", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "month", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "country", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil))

	files := []struct {
		name    string
		content string
	}{
		{"/dataset/2018/01/US/dat.json", `[
			{"region": "NY", "model": "3", "sales": 742.0},
			{"region": "NY", "model": "S", "sales": 304.125},
			{"region": "NY", "model": "X", "sales": 136.25},
			{"region": "NY", "model": "Y", "sales": 27.5}
		]`},
		{"/dataset/2018/01/CA/dat.json", `[
			{"region": "CA", "model": "3", "sales": 512},
			{"region": "CA", "model": "S", "sales": 978},
			{"region": "CA", "model": "X", "sales": 1.0},
			{"region": "CA", "model": "Y", "sales": 69}
		]`},
		{"/dataset/2019/01/US/dat.json", `[
			{"region": "QC", "model": "3", "sales": 273.5},
			{"region": "QC", "model": "S", "sales": 13},
			{"region": "QC", "model": "X", "sales": 54},
			{"region": "QC", "model": "Y", "sales": 21}
		]`},
		{"/dataset/2019/01/CA/dat.json", `[
			{"region": "QC", "model": "3", "sales": 152.25},
			{"region": "QC", "model": "S", "sales": 10},
			{"region": "QC", "model": "X", "sales": 42},
			{"region": "QC", "model": "Y", "sales": 37}
		]`},
		{"/dataset/.pesky", "garbage content"},
	}

	ete.expected = make([]string, 0, len(files))

	mockfs := &tools.MockFS{}
	for _, f := range files {
		ete.NoError(mockfs.CreateFile(f.name, []byte(f.content)))
		ete.expected = append(ete.expected, f.content)
	}
	ete.mockfs = mockfs
}

type JSONRecordBatchFileFormat struct {
	resolver func(*dataset.FileSource) *arrow.Schema
}

func (j *JSONRecordBatchFileFormat) Equals(f dataset.FileFormat) bool {
	return f == dataset.FileFormat(j)
}
func (j *JSONRecordBatchFileFormat) TypeName() string { return "json_record_batch" }
func (j *JSONRecordBatchFileFormat) IsSupported(f *dataset.FileSource) (bool, error) {
	return true, nil
}
func (j *JSONRecordBatchFileFormat) Inspect(f *dataset.FileSource) (*arrow.Schema, error) {
	return j.resolver(f), nil
}
func (j *JSONRecordBatchFileFormat) ScanFile(opts *dataset.ScanOptions, fragment *dataset.FileFragment) (dataset.ScanTaskIterator, error) {
	file, err := fragment.Source().Open()
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	schema, err := j.Inspect(fragment.Source())
	if err != nil {
		return nil, err
	}

	rec, err := types.RecordFromJSON(schema, data)
	if err != nil {
		return nil, err
	}

	return dataset.NewInMemoryFragment([]array.Record{rec}).Scan(opts)
}

// func (j *JSONRecordBatchFileFormat) ScanBatches(opts *dataset.ScanOptions, file *dataset.FileFragment) (dataset.RecordGenerator, error) {

// }
func (j *JSONRecordBatchFileFormat) MakeFragment(f *dataset.FileSource, opts ...dataset.MakeFragmentOption) (*dataset.FileFragment, error) {
	return dataset.NewFileFragmentWithOptions(f, j, opts...), nil
}

func (ete *EndToEndSuite) TestEndToEndSingle() {
	formatSchema := dataset.SchemaFromColumnNames(ete.sc, []string{"region", "model", "sales"})
	format := &JSONRecordBatchFileFormat{func(*dataset.FileSource) *arrow.Schema { return formatSchema }}

	fsd, err := dataset.NewFileSystemDatasetFromPaths([]string{
		"/dataset/2018/01/US/dat.json",
		"/dataset/2018/01/CA/dat.json",
		"/dataset/2019/01/US/dat.json",
		"/dataset/2019/01/CA/dat.json",
	},
		ete.mockfs, format,
		&dataset.FileSystemDatasetOptions{})

	ete.NoError(err)

	scanner, err := dataset.NewScanner(&ete.opts, fsd)
	ete.NoError(err)

	expected := expectedOutput(formatSchema, ete.expected[:len(ete.expected)-1])

	for rec := range scanner.ScanBatches() {
		ete.True(array.RecordEqual(expected[0], rec.RecordBatch))
		expected = expected[1:]
	}
}

func (ete *EndToEndSuite) TestDirectoryPartitioning() {
	formatSchema := dataset.SchemaFromColumnNames(ete.sc, []string{"region", "model", "sales"})
	format := &JSONRecordBatchFileFormat{func(*dataset.FileSource) *arrow.Schema { return formatSchema }}

	fsd, err := dataset.NewFileSystemDatasetFromPaths([]string{
		"/dataset/2018/01/US/dat.json",
		"/dataset/2018/01/CA/dat.json",
		"/dataset/2019/01/US/dat.json",
		"/dataset/2019/01/CA/dat.json",
	},
		ete.mockfs, format,
		&dataset.FileSystemDatasetOptions{
			InspectOpts:         dataset.InspectOptions{Fragments: 1},
			PartitionBaseDir:    "/dataset/",
			PartitioningFactory: dataset.NewDirectoryPartitioningFactory([]string{"year", "month", "country"})})
	ete.NoError(err)

	ete.NoError(dataset.SetProjectionNames(&ete.opts, []string{"sales", "model", "country"}))
	ete.setFilter(compute.And(compute.Equal(compute.NewFieldRef("year"), compute.NewLiteral(2019)), compute.Greater(compute.NewFieldRef("sales"), compute.NewLiteral(100.0))))
	scanner, err := dataset.NewScanner(&ete.opts, fsd)
	ete.NoError(err)

	expected, _ := types.RecordFromJSON(ete.opts.ProjectedSchema, []byte(`[
		{"model": "3", "sales": 273.5, "country": "US"},
		{"model": "3", "sales": 152.25, "country": "CA"}
	]`))

	rdr, _ := array.NewRecordReader(ete.opts.ProjectedSchema, []array.Record{expected.NewSlice(0, 1), expected.NewSlice(1, 2)})
	defer rdr.Release()

	ete.assertScannerEquals(rdr, scanner, true)
}
