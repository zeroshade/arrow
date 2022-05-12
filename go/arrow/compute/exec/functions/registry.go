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

package functions

import (
	"fmt"
	"sort"
	"sync"

	"github.com/apache/arrow/go/v9/arrow/compute"
)

type FunctionRegistry struct {
	nameToFunc sync.Map
	// nameToOpts sync.Map
}

func (fr *FunctionRegistry) AddFunction(fn compute.Function, allowOverwrite bool) error {
	// debug validate docstrings

	if allowOverwrite {
		fr.nameToFunc.Store(fn.Name(), fn)
		return nil
	}

	_, loaded := fr.nameToFunc.LoadOrStore(fn.Name(), fn)
	if loaded {
		return fmt.Errorf("already have a registered function with name: %s", fn.Name())
	}
	return nil
}

func (fr *FunctionRegistry) AddAlias(target, source string) error {
	fn, ok := fr.nameToFunc.Load(source)
	if !ok {
		return fmt.Errorf("no function registered with name: %s", source)
	}
	fr.nameToFunc.Store(target, fn)
	return nil
}

func (fr *FunctionRegistry) GetFunction(name string) (compute.Function, error) {
	if fn, ok := fr.nameToFunc.Load(name); ok {
		return fn.(compute.Function), nil
	}
	return nil, fmt.Errorf("no function registered with name: %s", name)
}

func (fr *FunctionRegistry) GetFunctionNames() []string {
	out := make([]string, 0)
	fr.nameToFunc.Range(func(key, value any) bool {
		out = append(out, key.(string))
		return true
	})
	sort.Strings(out)
	return out
}
