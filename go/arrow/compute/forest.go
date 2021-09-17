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

import "reflect"

type Forest struct {
	size             int
	descendantCounts []int
}

func NewForest(size int, isAncestor func(int, int) bool) (forest Forest) {
	forest.size = size
	forest.descendantCounts = make([]int, size)
	prntStack := make([]int, 0)

	for i := range forest.descendantCounts {
		for len(prntStack) != 0 {
			if isAncestor(prntStack[len(prntStack)-1], i) {
				break
			}

			// prntStack.back() has no more descendents; finalize count and pop
			forest.descendantCounts[prntStack[len(prntStack)-1]] = i - 1 - prntStack[len(prntStack)-1]
			prntStack = prntStack[:len(prntStack)-1]
		}
		prntStack = append(prntStack, i)
	}

	// finalize descendent count for anything left in the stack
	for len(prntStack) != 0 {
		forest.descendantCounts[prntStack[len(prntStack)-1]] = size - 1 - prntStack[len(prntStack)-1]
		prntStack = prntStack[:len(prntStack)-1]
	}
	return forest
}

func (forest Forest) Size() int { return forest.size }
func (forest Forest) Equals(other Forest) bool {
	return other.size == forest.size && reflect.DeepEqual(forest.descendantCounts, other.descendantCounts)
}

func (forest *Forest) Ref(i int) Ref { return Ref{forest, i} }

// Visit takes a pre and post function and visits the tree with eager pruning,
// a return of true from the pre function indicates a subtree should be visited,
// while false indicates the subtree should be skipped.
func (forest Forest) Visit(pre func(Ref) (bool, error), post func(Ref)) error {
	var back Ref

	prntStack := make([]Ref, 0)
	for i := 0; i < forest.size; i++ {
		ref := Ref{&forest, i}
		for len(prntStack) > 0 {
			if prntStack[len(prntStack)-1].IsAncestorOf(ref) {
				break
			}
			back, prntStack = prntStack[len(prntStack)-1], prntStack[:len(prntStack)-1]
			post(back)
		}

		visit, err := pre(ref)
		if err != nil {
			return err
		}

		if !visit {
			i += ref.NumDescendants()
			continue
		}

		prntStack = append(prntStack, ref)
	}

	return nil
}

type Ref struct {
	forest *Forest
	i      int
}

func (r Ref) NumDescendants() int { return r.forest.descendantCounts[r.i] }
func (r Ref) IsAncestorOf(ref Ref) bool {
	return r.i < ref.i && r.i+1+r.NumDescendants() > ref.i
}
func (r Ref) Ok() bool { return r.forest != nil }
