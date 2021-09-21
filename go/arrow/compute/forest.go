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
	"hash/maphash"
	"reflect"

	"github.com/apache/arrow/go/arrow"
)

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
	Idx    int
}

func (r Ref) NumDescendants() int { return r.forest.descendantCounts[r.Idx] }
func (r Ref) IsAncestorOf(ref Ref) bool {
	return r.Idx < ref.Idx && r.Idx+1+r.NumDescendants() > ref.Idx
}
func (r Ref) Ok() bool { return r.forest != nil }

type Encoded struct {
	Index     *int
	Guarantee []int32
}

type EncodedList []Encoded

func (e EncodedList) Len() int      { return len(e) }
func (e EncodedList) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

type ByGuarantee struct{ EncodedList }

func (b ByGuarantee) Less(i, j int) bool {
	g1 := b.EncodedList[i].Guarantee
	g2 := b.EncodedList[j].Guarantee
	for i := 0; i < min(len(g1), len(g2)); i++ {
		switch {
		case g1[i] < g2[i]:
			return true
		case g1[i] > g2[i]:
			return false
		}
	}

	if len(g1) != len(g2) {
		return len(g1) < len(g2)
	}

	if b.EncodedList[i].Index != nil {
		return false
	}

	return b.EncodedList[j].Index != nil
}

func IsAncestor(encoded []Encoded) func(int, int) bool {
	return func(l, r int) bool {
		if encoded[l].Index != nil {
			// leaf-level object (e.g. a Fragment): not an ancestor
			return false
		}

		ancestor := encoded[l].Guarantee
		descendant := encoded[r].Guarantee
		if len(descendant) >= len(ancestor) {
			return reflect.DeepEqual(ancestor, descendant[:len(ancestor)])
		}
		return false
	}
}

type Subtree struct {
	exprToCode map[uint64]struct {
		ex   Expression
		code int32
	}
	codeToExpr   []Expression
	subtreeExprs map[uint64]bool
	seed         maphash.Seed
}

func (s *Subtree) getOrInsert(expr Expression) int32 {
	next := int32(len(s.exprToCode))
	h := expr.Hash()
	it, ok := s.exprToCode[h]
	if !ok {
		s.codeToExpr = append(s.codeToExpr, expr)
		it.ex = expr
		it.code = next
		s.exprToCode[h] = it
	}
	return it.code
}

func (s *Subtree) encodeConjunctionMembers(expr Expression, codes []int32) []int32 {
	if call, ok := expr.(*Call); ok {
		if call.funcName == "and_kleene" {
			codes = s.encodeConjunctionMembers(call.GetArg(0), codes)
			return s.encodeConjunctionMembers(call.GetArg(1), codes)
		}
	}
	return append(codes, s.getOrInsert(expr))
}

func (s *Subtree) GetSubtreeExpr(encodedSubtree Encoded) Expression {
	return s.codeToExpr[encodedSubtree.Guarantee[len(encodedSubtree.Guarantee)-1]]
}

func (s *Subtree) generateSubtrees(guarantee []int32, out []Encoded) []Encoded {
	var h maphash.Hash
	// h.SetSeed(s.seed)
	for len(guarantee) > 0 {
		h.Reset()
		h.Write(arrow.Int32Traits.CastToBytes(guarantee))
		hash := h.Sum64()
		if !s.subtreeExprs[hash] {
			out = append(out, Encoded{nil, guarantee})
			s.subtreeExprs[hash] = true
		}
		guarantee = guarantee[:len(guarantee)-1]
	}
	return out
}

func (s *Subtree) encodeOneGuarantee(index int, guarantee Expression, out []Encoded) []Encoded {
	encodedGuarantee := Encoded{&index, nil}
	encodedGuarantee.Guarantee = s.encodeConjunctionMembers(guarantee, make([]int32, 0))
	out = s.generateSubtrees(encodedGuarantee.Guarantee, out)
	return append(out, encodedGuarantee)
}

func (s *Subtree) EncodeGuarantees(get func(int) Expression, count int) []Encoded {
	if s.codeToExpr == nil {
		s.codeToExpr = make([]Expression, 0)
		s.subtreeExprs = make(map[uint64]bool)
		s.exprToCode = make(map[uint64]struct {
			ex   Expression
			code int32
		})
	}
	out := make([]Encoded, 0, count)
	for i := 0; i < count; i++ {
		out = s.encodeOneGuarantee(i, get(i), out)
	}
	return out
}
