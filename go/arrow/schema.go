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

package arrow

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/apache/arrow/go/arrow/internal/debug"
)

type Metadata struct {
	keys   []string
	values []string
}

func NewMetadata(keys, values []string) Metadata {
	if len(keys) != len(values) {
		panic("arrow: len mismatch")
	}

	n := len(keys)
	if n == 0 {
		return Metadata{}
	}

	md := Metadata{
		keys:   make([]string, n),
		values: make([]string, n),
	}
	copy(md.keys, keys)
	copy(md.values, values)
	return md
}

func MetadataFrom(kv map[string]string) Metadata {
	md := Metadata{
		keys:   make([]string, 0, len(kv)),
		values: make([]string, 0, len(kv)),
	}
	for k := range kv {
		md.keys = append(md.keys, k)
	}
	sort.Strings(md.keys)
	for _, k := range md.keys {
		md.values = append(md.values, kv[k])
	}
	return md
}

func (md Metadata) Len() int         { return len(md.keys) }
func (md Metadata) Keys() []string   { return md.keys }
func (md Metadata) Values() []string { return md.values }

func (md Metadata) String() string {
	o := new(strings.Builder)
	fmt.Fprintf(o, "[")
	for i := range md.keys {
		if i > 0 {
			fmt.Fprintf(o, ", ")
		}
		fmt.Fprintf(o, "%q: %q", md.keys[i], md.values[i])
	}
	fmt.Fprintf(o, "]")
	return o.String()
}

// FindKey returns the index of the key-value pair with the provided key name,
// or -1 if such a key does not exist.
func (md Metadata) FindKey(k string) int {
	for i, v := range md.keys {
		if v == k {
			return i
		}
	}
	return -1
}

func (md Metadata) clone() Metadata {
	if len(md.keys) == 0 {
		return Metadata{}
	}

	o := Metadata{
		keys:   make([]string, len(md.keys)),
		values: make([]string, len(md.values)),
	}
	copy(o.keys, md.keys)
	copy(o.values, md.values)

	return o
}

func (md Metadata) sortedIndices() []int {
	idxes := make([]int, len(md.keys))
	for i := range idxes {
		idxes[i] = i
	}

	sort.Slice(idxes, func(i, j int) bool {
		return md.keys[idxes[i]] < md.keys[idxes[j]]
	})
	return idxes
}

func (md Metadata) Equal(rhs Metadata) bool {
	if md.Len() != rhs.Len() {
		return false
	}

	idxes := md.sortedIndices()
	rhsIdxes := rhs.sortedIndices()
	for i := range idxes {
		j := idxes[i]
		k := rhsIdxes[i]
		if md.keys[j] != rhs.keys[k] || md.values[j] != rhs.values[k] {
			return false
		}
	}
	return true
}

// Schema is a sequence of Field values, describing the columns of a table or
// a record batch.
type Schema struct {
	fields []Field
	index  map[string][]int
	meta   Metadata
}

// NewSchema returns a new Schema value from the slice of fields and metadata.
//
// NewSchema panics if there is a field with an invalid DataType.
func NewSchema(fields []Field, metadata *Metadata) *Schema {
	sc := &Schema{
		fields: make([]Field, 0, len(fields)),
		index:  make(map[string][]int, len(fields)),
	}
	if metadata != nil {
		sc.meta = metadata.clone()
	}
	for i, field := range fields {
		if field.Type == nil {
			panic("arrow: field with nil DataType")
		}
		sc.fields = append(sc.fields, field)
		sc.index[field.Name] = append(sc.index[field.Name], i)
	}
	return sc
}

func (sc *Schema) Metadata() Metadata { return sc.meta }
func (sc *Schema) Fields() []Field    { return sc.fields }
func (sc *Schema) Field(i int) Field  { return sc.fields[i] }

func (sc *Schema) FieldsByName(n string) ([]Field, bool) {
	indices, ok := sc.index[n]
	if !ok {
		return nil, ok
	}
	fields := make([]Field, 0, len(indices))
	for _, v := range indices {
		fields = append(fields, sc.fields[v])
	}
	return fields, ok
}

// FieldIndices returns the indices of the named field or nil.
func (sc *Schema) FieldIndices(n string) []int {
	return sc.index[n]
}

func (sc *Schema) HasField(n string) bool { return len(sc.FieldIndices(n)) > 0 }
func (sc *Schema) HasMetadata() bool      { return len(sc.meta.keys) > 0 }

// Equal returns whether two schema are equal.
// Equal does not compare the metadata.
func (sc *Schema) Equal(o *Schema) bool {
	switch {
	case sc == o:
		return true
	case sc == nil || o == nil:
		return false
	case len(sc.fields) != len(o.fields):
		return false
	}

	for i := range sc.fields {
		if !sc.fields[i].Equal(o.fields[i]) {
			return false
		}
	}
	return true
}

func (s *Schema) String() string {
	o := new(strings.Builder)
	fmt.Fprintf(o, "schema:\n  fields: %d\n", len(s.Fields()))
	for i, f := range s.Fields() {
		if i > 0 {
			o.WriteString("\n")
		}
		fmt.Fprintf(o, "    - %v", f)
	}
	if meta := s.Metadata(); meta.Len() > 0 {
		fmt.Fprintf(o, "\n  metadata: %v", meta)
	}
	return o.String()
}

func (s *Schema) HasDistinctFieldNames() bool {
	for _, v := range s.index {
		if len(v) > 1 {
			return false
		}
	}
	return true
}

type mergeCfg struct {
	promoteNullability bool
}

type MergeOption func(*mergeCfg)

func WithPromoteNullability(v bool) MergeOption {
	return func(cfg *mergeCfg) {
		cfg.promoteNullability = v
	}
}

func UnifySchemas(schemas []*Schema, opts ...MergeOption) (*Schema, error) {
	if len(schemas) == 0 {
		return nil, errors.New("must provide at least one schema to unify")
	}

	if !schemas[0].HasDistinctFieldNames() {
		return nil, errors.New("can't unify schema with duplicate field names")
	}

	bldr := NewSchemaBuilderFromSchema(schemas[0], ConflictMerge, opts...)
	if len(schemas) == 1 {
		return bldr.Finish(), nil
	}

	for _, s := range schemas[1:] {
		if !s.HasDistinctFieldNames() {
			return nil, errors.New("can't unify schema with duplicate field names")
		}
		if err := bldr.AddSchema(s); err != nil {
			return nil, err
		}
	}
	return bldr.Finish(), nil
}

type ConflictPolicy int8

const (
	ConflictAppend ConflictPolicy = iota
	ConflictIgnore
	ConflictReplace
	ConflictMerge
	ConflictError
)

var DefaultConflictPolicy = ConflictAppend

type SchemaBuilder struct {
	fields         []Field
	nameToIndex    map[string][]int
	metadata       Metadata
	Policy         ConflictPolicy
	fieldMergeOpts []MergeOption
}

type SchemaBuilderOption func(*SchemaBuilder)

func NewSchemaBuilder(policy ConflictPolicy, fieldMergeOpts ...MergeOption) *SchemaBuilder {
	return &SchemaBuilder{Policy: policy, fieldMergeOpts: fieldMergeOpts, nameToIndex: make(map[string][]int), fields: make([]Field, 0)}
}

func NewSchemaBuilderFromSchema(s *Schema, policy ConflictPolicy, fieldMergeOpts ...MergeOption) *SchemaBuilder {
	ret := NewSchemaBuilder(policy, fieldMergeOpts...)
	if s.HasMetadata() {
		ret.metadata = s.meta.clone()
	}

	ret.fields = make([]Field, len(s.fields))
	copy(ret.fields, s.fields)
	for k, v := range s.index {
		ret.nameToIndex[k] = v
	}
	return ret
}

const (
	notFound       = -1
	duplicateFound = -2
)

func lookupNameIndex(nameToIndex map[string][]int, name string) int {
	idxes, ok := nameToIndex[name]
	if !ok || len(idxes) == 0 {
		return notFound
	}

	if len(idxes) > 1 {
		return duplicateFound
	}

	return idxes[0]
}

func (sb *SchemaBuilder) appendField(f Field) {
	idxes := sb.nameToIndex[f.Name]
	if idxes == nil {
		idxes = make([]int, 0, 1)
	}
	idxes = append(idxes, len(sb.fields))
	sb.fields = append(sb.fields, f)
	sb.nameToIndex[f.Name] = idxes
}

func (sb *SchemaBuilder) AddField(f Field) error {
	if sb.Policy == ConflictAppend {
		sb.appendField(f)
		return nil
	}

	i := lookupNameIndex(sb.nameToIndex, f.Name)
	switch {
	case i == notFound:
		sb.appendField(f)
	case sb.Policy == ConflictIgnore:
	case sb.Policy == ConflictError:
		return errors.New("duplicate found, policy dictates to treat as an error")
	case i == duplicateFound:
		// cannot merge/replace when there's already more than one field
		// in the builder because we can't decide which to merge/replace
		return fmt.Errorf("cannot merge field %s more than one field with same name exists", f.Name)
	case sb.Policy == ConflictReplace:
		sb.fields[i] = f
	case sb.Policy == ConflictMerge:
		var err error
		sb.fields[i], err = sb.fields[i].MergeWith(f, sb.fieldMergeOpts...)
		if err != nil {
			return err
		}
	}

	return nil
}

func (sb *SchemaBuilder) AddFields(fields []Field) error {
	for _, f := range fields {
		if err := sb.AddField(f); err != nil {
			return err
		}
	}
	return nil
}

func (sb *SchemaBuilder) AddSchema(s *Schema) error {
	debug.Assert(s != nil || len(s.fields) == 0, "addschema cannot recieve nil schema or no fields")
	return sb.AddFields(s.fields)
}

func (sb *SchemaBuilder) AddSchemas(schemas []*Schema) error {
	for _, s := range schemas {
		if err := sb.AddSchema(s); err != nil {
			return err
		}
	}
	return nil
}

func (sb *SchemaBuilder) AddMetadata(metadata Metadata) {
	sb.metadata = metadata.clone()
}

func (sb *SchemaBuilder) Reset() {
	sb.fields = make([]Field, 0)
	sb.nameToIndex = make(map[string][]int)
	sb.metadata = NewMetadata(nil, nil)
}

func (sb *SchemaBuilder) Finish() *Schema {
	return &Schema{fields: sb.fields, index: sb.nameToIndex, meta: sb.metadata}
}

func MergeSchemas(schemas []*Schema, policy ConflictPolicy) (*Schema, error) {
	bldr := SchemaBuilder{Policy: policy}
	if err := bldr.AddSchemas(schemas); err != nil {
		return nil, err
	}
	return bldr.Finish(), nil
}

func SchemasAreCompatible(schemas []*Schema, policy ConflictPolicy) error {
	_, err := MergeSchemas(schemas, policy)
	return err
}
