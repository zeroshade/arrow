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

package exec_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/apache/arrow/go/v9/arrow"
	"github.com/apache/arrow/go/v9/arrow/array"
	"github.com/apache/arrow/go/v9/arrow/bitutil"
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec"
	"github.com/apache/arrow/go/v9/arrow/internal/testing/gen"
	"github.com/apache/arrow/go/v9/arrow/memory"
	"golang.org/x/exp/constraints"
)

const seed = 0x94378165

func genArray[T constraints.Integer](rg *gen.RandomArrayGenerator, dt arrow.DataType, size int64, prob float64, min, max T) arrow.Array {
	buffers := make([]*memory.Buffer, 2)
	nullCount := int64(0)

	buffers[0] = memory.NewResizableBuffer(rg.Alloc())
	buffers[0].Resize(int(bitutil.BytesForBits(size)))
	defer buffers[0].Release()
	nullCount = rg.GenerateBitmap(buffers[0].Bytes(), size, prob)

	buffers[1] = memory.NewResizableBuffer(rg.Alloc())
	buffers[1].Resize(int(size) * int(unsafe.Sizeof(T(0))))
	defer buffers[1].Release()

	dist := rand.New(rand.NewSource(seed))
	out := unsafe.Slice((*T)(unsafe.Pointer(&buffers[1].Bytes()[0])), size)
	for i := int64(0); i < size; i++ {
		out[i] = T(dist.Intn(int(max)-int(min+1))) + min
	}

	data := array.NewData(dt, int(size), buffers, nil, int(nullCount), 0)
	defer data.Release()
	return array.MakeFromData(data)
}

func benchmarkNumericCast[T constraints.Integer](b *testing.B, options compute.CastOptions, size int, nulls float64, fromType, toType arrow.DataType, min, max T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())

	rg := gen.NewRandomArrayGenerator(seed, mem)
	arr := genArray(&rg, fromType, int64(size), nulls, min, max)
	in := compute.NewDatum(arr)
	b.Cleanup(func() {
		arr.Release()
		in.Release()
		mem.AssertSize(b, 0)
	})

	ctx := compute.SetExecCtx(context.Background(), exec.DefaultExecCtx())
	b.ResetTimer()
	b.SetBytes(int64(size) * int64(unsafe.Sizeof(T(0))))
	for i := 0; i < b.N; i++ {
		out, err := exec.CastTo(ctx, in, toType, options)
		if err != nil {
			b.Error(err)
			b.FailNow()
		}
		out.Release()
	}
}

func BenchmarkCastInt64ToInt32(b *testing.B) {
	for _, safe := range []bool{true, false} {
		for _, sz := range []int{1.5 * 1024 * 1024} {
			size := sz / 8
			for _, nullprob := range []float64{1, .9, .5, .1, 0} {
				b.Run(fmt.Sprintf("safe=%t;sz=%d;nulls=%f", safe, sz, nullprob), func(b *testing.B) {
					benchmarkNumericCast[int64](b, *compute.DefaultCastOptions(safe), size, nullprob, arrow.PrimitiveTypes.Int64, arrow.PrimitiveTypes.Int32, math.MinInt32, math.MaxInt32)
				})
			}
		}
	}
}

func BenchmarkCastUint32ToInt32(b *testing.B) {
	for _, safe := range []bool{true, false} {
		for _, sz := range []int{1.5 * 1024 * 1024} {
			size := sz / 4
			for _, nullprob := range []float64{1, .9, .5, .1, 0} {
				b.Run(fmt.Sprintf("safe=%t;sz=%d;nulls=%f", safe, sz, nullprob), func(b *testing.B) {
					benchmarkNumericCast[uint32](b, *compute.DefaultCastOptions(safe), size, nullprob, arrow.PrimitiveTypes.Uint32, arrow.PrimitiveTypes.Int32, 0, math.MaxInt32)
				})
			}
		}
	}
}

func BenchmarkCastInt64ToFloat64(b *testing.B) {
	for _, safe := range []bool{true, false} {
		for _, sz := range []int{1.5 * 1024 * 1024} {
			size := sz / 8
			for _, nullprob := range []float64{1, .9, .5, .1, 0} {
				b.Run(fmt.Sprintf("safe=%t;sz=%d;nulls=%f", safe, sz, nullprob), func(b *testing.B) {
					benchmarkNumericCast[int64](b, *compute.DefaultCastOptions(safe), size, nullprob, arrow.PrimitiveTypes.Int64, arrow.PrimitiveTypes.Float64, 0, 1000)
				})
			}
		}
	}
}

// func BenchmarkCastFloat64ToInt32(b *testing.B) {
// 	for _, safe := range []bool{true, false} {
// 		for _, sz := range []int{1.5 * 1024 * 1024} {
// 			size := sz / 8
// 			for _, nullprob := range []float64{1, .9, .5, .1, 0} {
// 				b.Run(fmt.Sprintf("safe=%t;sz=%d;nulls=%f", safe, sz, nullprob), func(b *testing.B) {
// 					benchmarkNumericCast[int64](b, *compute.DefaultCastOptions(safe), size, nullprob, arrow.PrimitiveTypes.Float64, arrow.PrimitiveTypes.Int32, -1000, 1000)
// 				})
// 			}
// 		}
// 	}
// }
