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
	"github.com/pkg/errors"
)

type unionImpl struct {
	children []*Dataset
	schema   *arrow.Schema
}

func NewUnionDataset(schema *arrow.Schema, children []*Dataset) (*Dataset, error) {
	for _, c := range children {
		if !c.Schema().Equal(schema) {
			return nil, errors.Wrapf(TypeError, "child dataset had schema %s, but the union schema is %s", c.Schema(), schema)
		}
	}
	return &Dataset{&unionImpl{children, schema}, compute.NewLiteral(true)}, nil
}

func (u *unionImpl) Schema() *arrow.Schema { return u.schema }
func (u *unionImpl) TypeName() string      { return "union" }

func (u *unionImpl) getFragmentsImpl(predicate compute.BoundExpression) FragmentIterator {
	return GetFragmentsFromDatasets(u.children, predicate)
}

func (u *unionImpl) ReplaceSchema(schema *arrow.Schema) (*Dataset, error) {
	newChildren := make([]*Dataset, len(u.children))
	for i, c := range u.children {
		child, err := c.ReplaceSchema(schema)
		if err != nil {
			return nil, err
		}
		newChildren[i] = child
	}
	return NewUnionDataset(schema, newChildren)
}
