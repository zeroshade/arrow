//go:build ccalloc || ccexec
// +build ccalloc ccexec

package compute_test

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/apache/arrow/go/v7/arrow"
	"github.com/apache/arrow/go/v7/arrow/array"
	"github.com/apache/arrow/go/v7/arrow/compute"
	"github.com/apache/arrow/go/v7/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var rd *rand.Rand

func init() {
	rd = rand.New(rand.NewSource(42))
}

func buildChunk(nrows, chunk int, f arrow.Field, b array.Builder) {
	var add func(i int)
	switch f.Type {
	case arrow.PrimitiveTypes.Int64:
		add = func(i int) { b.(*array.Int64Builder).Append(int64(i + (chunk * nrows) + 1)) }
	case arrow.PrimitiveTypes.Float64:
		add = func(i int) { b.(*array.Float64Builder).Append(float64((i + (chunk * nrows) + 1)) * 0.25) }
	case arrow.BinaryTypes.String:
		if strings.HasSuffix(f.Name, "-rand-dist") {
			add = func(i int) {
				b.(*array.StringBuilder).Append([]string{"Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"}[rd.Intn(7)])
			}
		} else {
			add = func(i int) { b.(*array.StringBuilder).Append(fmt.Sprintf("row%dchunk%d", i+(chunk*nrows), chunk)) }
		}
	}

	for i := 0; i < nrows; i++ {
		add(i)
	}
}

func randomTable(mem memory.Allocator, numCols, numRows int) array.Table {
	fields := make([]arrow.Field, numCols)
	for i := 0; i < numCols; i++ {
		switch {
		case i%7 == 0:
			fields[i] = arrow.Field{Name: fmt.Sprintf("%d-rand-dist", i), Type: arrow.BinaryTypes.String, Nullable: false}
		case i%3 == 0:
			fields[i] = arrow.Field{Name: fmt.Sprintf("%d-int64", i), Type: arrow.PrimitiveTypes.Int64, Nullable: false}
		case i%2 == 0:
			fields[i] = arrow.Field{Name: fmt.Sprintf("%d-real", i), Type: arrow.PrimitiveTypes.Float64, Nullable: false}
		default:
			fields[i] = arrow.Field{Name: "string", Type: arrow.BinaryTypes.String, Nullable: false}
		}
	}

	sc := arrow.NewSchema(fields, nil)
	numChunks := 2
	if numRows%2 != 0 {
		numChunks = 3
	}

	builder := array.NewRecordBuilder(mem, sc)
	defer builder.Release()

	recs := make([]array.Record, 0)

	rowsInChunk := numRows / numChunks
	var r int
	for r = 0; r+rowsInChunk < numRows; r += rowsInChunk {
		for i, f := range sc.Fields() {
			buildChunk(rowsInChunk, len(recs), f, builder.Field(i))
		}
		rec := builder.NewRecord()
		defer rec.Release()
		recs = append(recs, rec)
	}

	if r < numRows {
		rowsInChunk = numRows - r
		for i, f := range sc.Fields() {
			buildChunk(rowsInChunk, len(recs), f, builder.Field(i))
		}
		rec := builder.NewRecord()
		defer rec.Release()
		recs = append(recs, rec)
	}

	return array.NewTableFromRecords(sc, recs)
}

func TestSortIndices(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewCgoArrowAllocator())
	defer mem.AssertSize(t, 0)

	tbl := randomTable(mem, 5, 1000000)
	defer tbl.Release()

	datum := compute.NewDatum(tbl)
	defer datum.Release()

	output, err := compute.CallFunction(context.Background(), mem, "sort_indices",
		[]compute.Datum{datum},
		&compute.SortOptions{
			SortKeys: []compute.SortKey{{Target: compute.FieldRefName("0-rand-dist")}}})
	require.NoError(t, err)
	defer output.Release()

	// datum2 := compute.NewDatum(tbl)
	// defer datum2.Release()

	// result, err := compute.CallFunction(context.Background(), mem, "take", []compute.Datum{datum2, output}, &compute.TakeOptions{BoundsCheck: true})
	// assert.NoError(t, err)
	// defer result.Release()

	// fmt.Println("foobar")
	// assert.Equal(t, compute.KindTable, result.Kind())
	// fmt.Println(result.String())
	// fmt.Println(result.(*compute.TableDatum))

	assert.Equal(t, compute.KindArray, output.Kind())
	data := output.(*compute.ArrayDatum).MakeArray()
	defer data.Release()

	indices := data.(*array.Uint64).Uint64Values()
	require.Equal(t, 1000000, len(indices))
	col := tbl.Column(0)
	for i, idx := range indices[:len(indices)-1] {
		next := indices[i+1]

		assert.LessOrEqual(t, findVal(idx, col), findVal(next, col))
	}
}

func BenchmarkSortedIndices(b *testing.B) {
	mem := memory.NewCheckedAllocator(memory.NewCgoArrowAllocator())

	tbl := randomTable(mem, 5, 1000000)
	defer tbl.Release()

	datum := compute.NewDatum(tbl)
	defer datum.Release()

	ctx := context.Background()
	funcName := "sort_indices"
	args := []compute.Datum{datum}
	opts := &compute.SortOptions{
		SortKeys: []compute.SortKey{{Target: compute.FieldRefName("0-rand-dist")}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := compute.CallFunction(ctx, mem, funcName, args, opts)
		require.NoError(b, err)
		out.Release()
	}
}

func findVal(i uint64, c *array.Column) interface{} {
	var (
		absPos   uint64
		offset   uint64
		chunkIdx int
	)

	for idx, chunk := range c.Data().Chunks() {
		chunkIdx = idx
		if absPos >= i {
			break
		}

		chunkLen := uint64(chunk.Len())
		if absPos+chunkLen > i {
			offset = i - absPos
			break
		}
		absPos += chunkLen
	}

	chunk := c.Data().Chunks()[chunkIdx]
	switch chunk.DataType().ID() {
	case arrow.STRING:
		return chunk.(*array.String).Value(int(offset))
	}
	panic("not found")
}
