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
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"unicode/utf8"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/compute"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/apache/arrow/go/arrow/scalar"
)

type SegmentEncoding int8

const (
	SegmentNone SegmentEncoding = iota
	SegmentURI
)

type PartitionedBatches struct {
	Batches     []array.Record
	Expressions []compute.Expression
}

type Partitioning interface {
	TypeName() string
	Partition(array.Record) (PartitionedBatches, error)
	Parse(path string) (compute.Expression, error)
	Format(compute.Expression) (string, error)
	Schema() *arrow.Schema
}

type DefaultPartitioning struct{}

func (DefaultPartitioning) TypeName() string { return "default" }
func (DefaultPartitioning) Parse(string) (compute.Expression, error) {
	return compute.NewLiteral(true), nil
}
func (d DefaultPartitioning) Format(compute.Expression) (string, error) {
	return "", fmt.Errorf("formatting paths from %s Partitioning not implemented", d.TypeName())
}
func (DefaultPartitioning) Partition(batch array.Record) (PartitionedBatches, error) {
	return PartitionedBatches{[]array.Record{batch}, []compute.Expression{compute.NewLiteral(true)}}, nil
}
func (DefaultPartitioning) Schema() *arrow.Schema { return &arrow.Schema{} }

type PartitioningOptions struct {
	Schema          *arrow.Schema
	SegmentEncoding SegmentEncoding
}

type PartitioningFactory interface {
	Make(schema *arrow.Schema) (Partitioning, error)
	Inspect(paths []string) (*arrow.Schema, error)
}

func stripPrefixFilename(path, prefix string) string {
	baseLess, err := filepath.Rel(prefix, path)
	if err != nil {
		baseLess = path
	}

	return filepath.Dir(baseLess)
}

func stripPrefixAndFilenames(files []*FileSource, prefix string) (out []string) {
	out = make([]string, len(files))
	for i, f := range files {
		out[i] = stripPrefixFilename(f.Path(), prefix)
	}
	return
}

type PartitionKey struct {
	Name  string
	Value *string
}

type KeyValuePartionOptions struct {
	SegmentEncoding SegmentEncoding
}

type KeyValuePartitioning struct {
	impl KeyValuePartitionType

	schema  *arrow.Schema
	options KeyValuePartionOptions
}

func (KeyValuePartitioning) TypeName() string        { return "key-value" }
func (k KeyValuePartitioning) Schema() *arrow.Schema { return k.schema }
func (k *KeyValuePartitioning) convertKey(key PartitionKey) (compute.Expression, error) {
	match, err := compute.NewFieldNameRef(key.Name).FindOneOrNone(k.schema)
	if err != nil {
		return nil, err
	}

	if len(match) == 0 {
		return compute.NewLiteral(true), nil
	}

	idx := match[0]
	field := k.schema.Field(idx)

	if key.Value == nil {
		return compute.IsNull(compute.NewFieldRef(field.Name), true), nil
	}

	converted, err := scalar.ParseScalar(field.Type, *key.Value)
	if err != nil {
		return nil, err
	}

	return compute.Equal(compute.NewFieldRef(field.Name), compute.NewLiteral(converted)), nil
}

func (k *KeyValuePartitioning) Format(compute.Expression) (string, error) { return "", nil }

type KeyValuePartitionType interface {
	ParseKeys(path string, schema *arrow.Schema, enc SegmentEncoding) ([]PartitionKey, error)
	FormatValues(values []scalar.Scalar, schema *arrow.Schema) (string, error)
}

func (k *KeyValuePartitioning) Partition(batch array.Record) (PartitionedBatches, error) {
	return PartitionedBatches{}, errors.New("not implemented")
}

func (k *KeyValuePartitioning) Parse(path string) (compute.Expression, error) {
	exprs := make([]compute.Expression, 0)
	parsed, err := k.impl.ParseKeys(path, k.schema, k.options.SegmentEncoding)
	if err != nil {
		return nil, err
	}

	literalTrue := compute.NewLiteral(true)

	for _, key := range parsed {
		expr, err := k.convertKey(key)
		if err != nil {
			return nil, err
		}

		if expr.Equals(literalTrue) {
			continue
		}

		exprs = append(exprs, expr)
	}

	return compute.And(exprs...), nil
}

func getSplitPath(path string) []string {
	out := make([]string, 0)
	for len(path) > 0 && path != "." && path != string(filepath.Separator) {
		out = append(out, filepath.Base(path))
		path = filepath.Dir(path)
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

type directoryPartitioningImpl struct{}

func (directoryPartitioningImpl) ParseKeys(path string, schema *arrow.Schema, enc SegmentEncoding) ([]PartitionKey, error) {
	keys := make([]PartitionKey, 0)

	for i, segment := range getSplitPath(path) {
		if i >= len(schema.Fields()) {
			break
		}

		switch enc {
		case SegmentNone:
			if !utf8.ValidString(segment) {
				return nil, fmt.Errorf("partition segment was not valid utf-8: %s", segment)
			}
			seg := segment
			keys = append(keys, PartitionKey{Name: schema.Field(i).Name, Value: &seg})
		case SegmentURI:
			decoded, err := url.PathUnescape(segment)
			if err != nil {
				return nil, err
			}
			keys = append(keys, PartitionKey{Name: schema.Field(i).Name, Value: &decoded})
		default:
			return nil, fmt.Errorf("unknown segment encoding: %d", enc)
		}
	}
	return keys, nil
}

func nextValid(values []scalar.Scalar, firstNull int) *int {
	i := 0
	for ; i < len(values); i++ {
		if values[i] != nil {
			break
		}
	}

	if i == len(values) {
		return nil
	}

	return &i
}

func (directoryPartitioningImpl) FormatValues(values []scalar.Scalar, schema *arrow.Schema) (string, error) {
	segments := make([]string, len(schema.Fields()))

	for i := range schema.Fields() {
		if values[i] != nil && values[i].IsValid() {
			segments[i] = values[i].String()
			continue
		}

		if badindex := nextValid(values, i); badindex != nil {
			return "", fmt.Errorf("no partition key for %s but a key was provided later for %s", schema.Field(i).Name, schema.Field(*badindex).Name)
		}

		break
	}

	return filepath.Join(segments...), nil
}

type memo struct {
	bldr *array.StringBuilder
	vals map[string]bool
}

type keyValuePartitionFactory struct {
	opts        PartitioningOptions
	nameToIndex map[string]int

	memos        []memo
	dictionaries []array.Interface
}

func (k *keyValuePartitionFactory) Reset() {
	k.nameToIndex = make(map[string]int)
	k.memos = make([]memo, 0)
}

func (k *keyValuePartitionFactory) getOrInsertField(name string) int {
	var (
		idx int
		ok  bool
	)
	if idx, ok = k.nameToIndex[name]; !ok {
		idx = len(k.nameToIndex)
		k.nameToIndex[name] = idx

		k.memos = append(k.memos, memo{array.NewStringBuilder(memory.DefaultAllocator), make(map[string]bool)})
	}
	return idx
}

func (k *keyValuePartitionFactory) insertRepr(fieldName string, repr *string) error {
	idx := k.getOrInsertField(fieldName)
	if repr != nil {
		return k.insertValue(idx, *repr)
	}
	return nil
}

func (k *keyValuePartitionFactory) insertValue(index int, repr string) error {
	if _, ok := k.memos[index].vals[repr]; !ok {
		memo := k.memos[index]
		memo.bldr.Append(repr)
		memo.vals[repr] = true
	}
	return nil
}

func (k *keyValuePartitionFactory) doInspect() (*arrow.Schema, error) {
	fields := make([]arrow.Field, len(k.nameToIndex))
	if k.opts.Schema != nil {
		requestedSize := len(k.opts.Schema.Fields())
		inferred := len(fields)
		if inferred != requestedSize {
			return nil, fmt.Errorf("requested schema has %d fields but only %d fields were detected", requestedSize, inferred)
		}
	}

	ctx := compute.ExecContext(context.Background())
	defer compute.ReleaseContext(ctx)

	k.dictionaries = make([]array.Interface, len(fields))
	for name, index := range k.nameToIndex {
		memo := k.memos[index]
		reprs := memo.bldr.NewStringArray()
		memo.bldr.Release()

		if reprs.Len() == 0 {
			return nil, fmt.Errorf("no non-null segments were avaialble for the field '%s'; couldn't infer type", name)
		}

		var (
			curField arrow.Field
			dictVals array.Interface
		)

		if k.opts.Schema != nil {
			curField = k.opts.Schema.Field(index)
			castTarget := curField.Type
			expr := compute.NewCall("cast", []compute.Expression{compute.NewFieldRef(name)}, compute.NewFunctionOption(compute.CastOptions{ToType: castTarget}))

			rb := array.NewRecord(arrow.NewSchema([]arrow.Field{curField}, nil), []array.Interface{reprs}, -1)
			defer rb.Release()

			out, err := compute.ExecuteScalarExpr(ctx, memory.DefaultAllocator, rb, expr)
			if err != nil {
				return nil, err
			}
			defer out.Release()
			// do casting
			dictVals = out.(*compute.ArrayDatum).MakeArray()
		} else {
			expr := compute.NewCall("cast", []compute.Expression{compute.NewFieldRef(name)}, compute.NewFunctionOption(compute.CastOptions{ToType: arrow.PrimitiveTypes.Int32}))
			rb := array.NewRecord(arrow.NewSchema([]arrow.Field{{Name: name, Type: arrow.BinaryTypes.String}}, nil), []array.Interface{reprs}, -1)
			defer rb.Release()

			out, err := compute.ExecuteScalarExpr(ctx, memory.DefaultAllocator, rb, expr)
			if err != nil {
				dictVals = reprs
			} else {
				defer out.Release()
				dictVals = out.(*compute.ArrayDatum).MakeArray()
			}

			// try casting to int32, otherwise bail and just use strings
			curField = arrow.Field{Name: name, Type: dictVals.DataType()}
		}
		fields[index] = curField
		k.dictionaries[index] = dictVals
	}
	k.Reset()
	return arrow.NewSchema(fields, nil), nil
}

type DirectoryPartitioningFactory struct {
	keyValuePartitionFactory
	fieldNames []string
}

func NewDirectoryPartitioningFactory(fieldnames []string) *DirectoryPartitioningFactory {
	ret := &DirectoryPartitioningFactory{
		fieldNames: fieldnames,
		keyValuePartitionFactory: keyValuePartitionFactory{
			opts: PartitioningOptions{SegmentEncoding: SegmentURI},
		},
	}
	ret.Reset()
	return ret
}

func (d *DirectoryPartitioningFactory) Reset() {
	d.keyValuePartitionFactory.Reset()
	for _, n := range d.fieldNames {
		d.getOrInsertField(n)
	}
}

func (DirectoryPartitioningFactory) TypeName() string { return "directory" }

func (d *DirectoryPartitioningFactory) Inspect(paths []string) (*arrow.Schema, error) {
	for _, p := range paths {
		for i, segment := range getSplitPath(p) {
			if i == len(d.fieldNames) {
				break
			}

			switch d.opts.SegmentEncoding {
			case SegmentNone:
				if !utf8.ValidString(segment) {
					return nil, fmt.Errorf("partition segment was not valid utf-8: %s", segment)
				}
				if err := d.insertValue(i, segment); err != nil {
					return nil, err
				}
			case SegmentURI:
				decoded, err := url.PathUnescape(segment)
				if err != nil {
					return nil, err
				}
				if err := d.insertValue(i, decoded); err != nil {
					return nil, err
				}
			default:
				return nil, fmt.Errorf("unknown segment encoding: %d", d.opts.SegmentEncoding)
			}
		}
	}
	return d.doInspect()
}

func (d *DirectoryPartitioningFactory) Make(schema *arrow.Schema) (Partitioning, error) {
	for _, n := range d.fieldNames {
		if _, err := compute.NewFieldNameRef(n).FindOne(schema); err != nil {
			return nil, err
		}
	}

	out := SchemaFromColumnNames(schema, d.fieldNames)
	return &KeyValuePartitioning{
		directoryPartitioningImpl{},
		out,
		KeyValuePartionOptions{d.opts.SegmentEncoding},
	}, nil
}
