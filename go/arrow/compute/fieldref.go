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

package compute

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"golang.org/x/xerrors"
)

type FieldPath []int

func (f FieldPath) String() string {
	if len(f) == 0 {
		return "FieldPath(empty)"
	}

	var b strings.Builder
	b.WriteString("FieldPath(")
	for _, i := range f {
		fmt.Fprint(&b, i)
		b.WriteByte(' ')
	}
	ret := b.String()
	return ret[:len(ret)-1] + ")"
}

func getFields(typ arrow.DataType) []arrow.Field {
	switch dt := typ.(type) {
	case *arrow.ListType:
		return []arrow.Field{{Name: "item", Type: dt.Elem()}}
	case *arrow.FixedSizeListType:
		return []arrow.Field{{Name: "item", Type: dt.Elem()}}
	case *arrow.StructType:
		return dt.Fields()
	case *arrow.MapType:
		return dt.ValueType().Fields()
	default:
		return []arrow.Field{}
	}
}

func (f FieldPath) GetFieldFromSlice(fields []arrow.Field) (*arrow.Field, error) {
	if len(f) == 0 {
		return nil, xerrors.New("cannot traverse empty field path")
	}

	var (
		depth = 0
		out   *arrow.Field
	)
	for _, idx := range f {
		if len(fields) == 0 {
			return nil, xerrors.Errorf("trying to get child of type with no children")
		}

		if idx < 0 || idx >= len(fields) {
			return nil, xerrors.Errorf("index out of range. indices=%s", f[:depth+1])
		}

		out = &fields[idx]
		fields = getFields(out.Type)
		depth++
	}

	return out, nil
}

func getChildren(arr array.Interface) (ret []array.Interface) {
	switch arr := arr.(type) {
	case *array.Struct:
		ret = make([]array.Interface, arr.NumField())
		for i := 0; i < arr.NumField(); i++ {
			ret[i] = arr.Field(i)
		}
	case *array.List:
		ret = []array.Interface{arr.ListValues()}
	case *array.FixedSizeList:
		ret = []array.Interface{arr.ListValues()}
	case *array.Map:
		ret = []array.Interface{arr.ListValues()}
	}
	return
}

func (f FieldPath) getArray(arrs []array.Interface) (array.Interface, error) {
	if len(f) == 0 {
		return nil, xerrors.New("cannot traverse empty field path")
	}

	var (
		depth = 0
		out   array.Interface
	)
	for _, idx := range f {
		if len(arrs) == 0 {
			return nil, xerrors.Errorf("trying to get child of array list with no children")
		}

		if idx < 0 || idx >= len(arrs) {
			return nil, xerrors.Errorf("index out of range. indices=%s", f[:depth+1])
		}

		out = arrs[idx]
		arrs = getChildren(out)
		depth++
	}

	return out, nil
}

func (f FieldPath) GetField(field arrow.Field) (*arrow.Field, error) {
	return f.GetFieldFromType(field.Type)
}

func (f FieldPath) GetFieldFromType(typ arrow.DataType) (*arrow.Field, error) {
	return f.GetFieldFromSlice(getFields(typ))
}

func (f FieldPath) findAll(fields []arrow.Field) []FieldPath {
	_, err := f.GetFieldFromSlice(fields)
	if err == nil {
		return []FieldPath{f}
	}
	return []FieldPath{}
}

func (f FieldPath) GetColumn(batch array.Record) (array.Interface, error) {
	return f.getArray(batch.Columns())
}

func NewFieldRefFromDotPath(dotpath string) (out FieldRef, err error) {
	if len(dotpath) == 0 {
		return out, xerrors.New("dotpath was empty")
	}

	parseName := func() string {
		var name string
		for {
			idx := strings.IndexAny(dotpath, `\[.`)
			if idx == -1 {
				name += dotpath
				dotpath = ""
				break
			}

			if dotpath[idx] != '\\' {
				// subscript for a new fieldref
				name += dotpath[:idx]
				dotpath = dotpath[idx:]
				break
			}

			if len(dotpath) == idx+1 {
				// dotpath ends with backslash; consume it all
				name += dotpath
				dotpath = ""
				break
			}

			// append all characters before backslash, then the character which follows it
			name += dotpath[:idx]
			name += string(dotpath[idx+1])
			dotpath = dotpath[idx+2:]
		}
		return name
	}

	children := make([]FieldRef, 0)

	for len(dotpath) > 0 {
		subscript := dotpath[0]
		dotpath = dotpath[1:]
		switch subscript {
		case '.':
			// next element is a name
			children = append(children, FieldRef{fieldNameRef(parseName())})
		case '[':
			subend := strings.IndexFunc(dotpath, func(r rune) bool { return !unicode.IsDigit(r) })
			if subend == -1 || dotpath[subend] != ']' {
				return out, xerrors.Errorf("dot path '%s' contained an unterminated index", dotpath)
			}
			idx, _ := strconv.Atoi(dotpath[:subend])
			children = append(children, FieldRef{FieldPath{idx}})
			dotpath = dotpath[subend+1:]
		default:
			return out, xerrors.Errorf("dot path must begin with '[' or '.' got '%s'", dotpath)
		}
	}

	out.flatten(children)
	return
}

func NewFieldNameRef(name string) FieldRef {
	return FieldRef{fieldNameRef(name)}
}

type FieldRef struct {
	impl fieldRefImpl
}

func (f *FieldRef) IsName() bool {
	_, ok := f.impl.(fieldNameRef)
	return ok
}

func (f *FieldRef) IsFieldPath() bool {
	_, ok := f.impl.(FieldPath)
	return ok
}

func (f *FieldRef) IsNested() bool {
	switch impl := f.impl.(type) {
	case fieldNameRef:
		return false
	case FieldPath:
		return len(impl) > 1
	default:
		return true
	}
}

func (f *FieldRef) Name() string {
	n, _ := f.impl.(fieldNameRef)
	return string(n)
}

func (f *FieldRef) FieldPath() FieldPath {
	p, _ := f.impl.(FieldPath)
	return p
}

func (f *FieldRef) Equals(other FieldRef) bool {
	return reflect.DeepEqual(f.impl, other.impl)
}

func (f *FieldRef) flatten(children []FieldRef) {
	out := make([]FieldRef, 0, len(children))

	var populate func(fieldRefImpl)
	populate = func(refs fieldRefImpl) {
		switch r := refs.(type) {
		case fieldNameRef:
			out = append(out, FieldRef{r})
		case FieldPath:
			out = append(out, FieldRef{r})
		case fieldRefList:
			for _, c := range r {
				populate(c.impl)
			}
		}
	}

	populate(fieldRefList(children))

	if len(out) == 1 {
		f.impl = out[0].impl
	} else {
		f.impl = fieldRefList(out)
	}
}

func (f FieldRef) FindAll(fields []arrow.Field) []FieldPath {
	return f.impl.findAll(fields)
}

func (f FieldRef) FindAllField(field arrow.Field) []FieldPath {
	return f.impl.findAll(getFields(field.Type))
}

func (f FieldRef) FindOneOrNoneRecord(root array.Record) (FieldPath, error) {
	matches := f.FindAll(root.Schema().Fields())
	if len(matches) > 1 {
		return nil, xerrors.Errorf("multiple matches for %s in %s", f, root.Schema())
	}
	if len(matches) == 0 {
		return FieldPath{}, nil
	}
	return matches[0], nil
}

func (f FieldRef) GetAllColumns(root array.Record) ([]array.Interface, error) {
	out := make([]array.Interface, 0)
	for _, m := range f.FindAll(root.Schema().Fields()) {
		n, err := m.GetColumn(root)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, nil
}

func (f FieldRef) GetOneColumnOrNone(root array.Record) (array.Interface, error) {
	match, err := f.FindOneOrNoneRecord(root)
	if err != nil {
		return nil, err
	}
	if len(match) == 0 {
		return nil, nil
	}
	return match.GetColumn(root)
}

type fieldNameRef string

func (ref fieldNameRef) findAll(fields []arrow.Field) []FieldPath {
	out := []FieldPath{}
	for i, f := range fields {
		if f.Name == string(ref) {
			out = append(out, FieldPath{i})
		}
	}
	return out
}

type fieldRefList []FieldRef

type matches struct {
	prefixes []FieldPath
	refs     []*arrow.Field
}

func (m *matches) add(prefix, suffix FieldPath, fields []arrow.Field) {
	f, err := suffix.GetFieldFromSlice(fields)
	if err != nil {
		panic(err)
	}

	m.refs = append(m.refs, f)
	m.prefixes = append(m.prefixes, append(prefix, suffix...))
}

func (ref fieldRefList) findAll(fields []arrow.Field) []FieldPath {
	m := matches{}
	for _, list := range ref[0].FindAll(fields) {
		m.add(FieldPath{}, list, fields)
	}

	for _, r := range ref[1:] {
		next := matches{}
		for i, f := range m.refs {
			for _, match := range r.FindAllField(*f) {
				next.add(m.prefixes[i], match, getFields(f.Type))
			}
		}
		m = next
	}

	return m.prefixes
}

type fieldRefImpl interface {
	findAll(fields []arrow.Field) []FieldPath
}
