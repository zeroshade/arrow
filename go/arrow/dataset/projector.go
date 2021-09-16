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
	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/compute"
	"golang.org/x/xerrors"
)

func checkProjectable(from, to *arrow.Schema) error {
	for _, f := range to.Fields() {
		fromField, err := compute.NewFieldNameRef(f.Name).GetOneOrNone(from)
		if err != nil {
			return err
		}

		switch {
		case fromField == nil:
			if f.Nullable {
				continue
			}

			return xerrors.Errorf("%w: field %s is not nullable but has type %s in origin schema", TypeError, f, arrow.NULL)
		case fromField.Type.ID() == arrow.NULL:
			// promotion from null to any type is supported
			if f.Nullable {
				continue
			}

			return xerrors.Errorf("%w: field %s is not nullable but has type %s in origin schema", TypeError, f, arrow.Null)
		case !arrow.TypeEqual(fromField.Type, f.Type):
			return xerrors.Errorf("%w: fields had matching names but differing types: From: %s, To: %s", TypeError, fromField, f)
		case fromField.Nullable && !f.Nullable:
			return xerrors.Errorf("%w: field %s is not nullable but is not required in origin schema %s", TypeError, f, fromField)
		}
	}
	return nil
}

func SetProjection(opts *ScanOptions, projection compute.Expression) error {
	opts.Projection = compute.BindExpression(opts.Ctx, opts.Mem, projection, opts.DatasetSchema)
	typ, err := opts.Projection.Type()
	if err != nil {
		return err
	}

	if typ.ID() != arrow.STRUCT {
		return xerrors.Errorf("Projection %s cannot yield record batches", typ)
	}

	meta := opts.DatasetSchema.Metadata()
	opts.ProjectedSchema = arrow.NewSchema(typ.(*arrow.StructType).Fields(), &meta)
	return nil
}

func SetProjectionExprs(opts *ScanOptions, exprs []compute.Expression, names []string) error {
	projOpts := compute.MakeStructOptions{
		FieldNames:       names,
		FieldNullability: make([]bool, len(names)),
		FieldMetadata:    make([]*arrow.Metadata, len(names)),
	}
	for i, e := range exprs {
		if ref := e.FieldRef(); ref != nil {
			if ref.Name() == "" {
				return xerrors.New("nested field refs not implemented")
			}

			field, err := ref.GetOne(opts.DatasetSchema)
			if err != nil {
				return err
			}

			projOpts.FieldNullability[i] = field.Nullable
			projOpts.FieldMetadata[i] = &field.Metadata
		}
	}
	return SetProjection(opts, compute.NewCall("make_struct", exprs, compute.NewFunctionOption(projOpts)))
}

func SetProjectionNames(opts *ScanOptions, names []string) error {
	exprs := make([]compute.Expression, len(names))
	for i, n := range names {
		exprs[i] = compute.NewFieldRef(n)
	}
	return SetProjectionExprs(opts, exprs, names)
}

func SetFilter(opts *ScanOptions, filter compute.Expression) error {
	for _, ref := range compute.FieldsInExpression(filter) {
		if ref.Name() == "" {
			return xerrors.New("nested field refs not implemented")
		}

		if _, err := ref.FindOne(opts.DatasetSchema); err != nil {
			return err
		}
	}

	opts.Filter = compute.BindExpression(opts.Ctx, opts.Mem, filter, opts.DatasetSchema)
	return nil
}
