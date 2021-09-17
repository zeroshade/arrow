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

package scalar

import (
	"fmt"
	"math"
	"math/big"
	"reflect"
	"unsafe"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/decimal128"
	"github.com/apache/arrow/go/arrow/float16"
	"github.com/apache/arrow/go/arrow/internal/debug"
	"github.com/apache/arrow/go/arrow/memory"
	"golang.org/x/xerrors"
)

type Scalar interface {
	fmt.Stringer
	IsValid() bool
	DataType() arrow.DataType
	Validate() error
	ValidateFull() error
	CastTo(arrow.DataType) (Scalar, error)
	value() interface{}
	equals(Scalar) bool
	//TODO(zeroshade): approxEquals
}

func validateOptional(s *scalar, value interface{}, valueDesc string) error {
	if s.Valid && value == nil {
		return xerrors.Errorf("%s scalar is marked valid but doesn't have a %s", s.Type, valueDesc)
	}
	if !s.Valid && value != nil && !reflect.ValueOf(value).IsNil() {
		return xerrors.Errorf("%s scalar is marked null but has a %s", s.Type, valueDesc)
	}
	return nil
}

type scalar struct {
	Type  arrow.DataType
	Valid bool
}

func (s *scalar) String() string {
	if !s.Valid {
		return "null"
	}

	return "..."
}

func (s *scalar) IsValid() bool { return s.Valid }

func (s *scalar) Validate() error {
	if s.Type == nil {
		return xerrors.New("scalar lacks a type")
	}
	return nil
}

func (s *scalar) ValidateFull() error {
	return s.Validate()
}

func (s scalar) DataType() arrow.DataType { return s.Type }

type Null struct {
	scalar
}

// by the time we get here we already know that the rhs is the right type
func (n *Null) equals(s Scalar) bool {
	debug.Assert(s.DataType().ID() == arrow.NULL, "scalar null equals should only receive null")
	return true
}

func (n *Null) value() interface{} { return nil }

func (n *Null) CastTo(dt arrow.DataType) (Scalar, error) {
	return MakeNullScalar(dt), nil
}

func (n *Null) Validate() (err error) {
	err = n.scalar.Validate()
	if err != nil {
		return
	}
	if n.Valid {
		err = xerrors.New("null scalar should have Valid = false")
	}
	return
}

func (n *Null) ValidateFull() error { return n.Validate() }

var (
	ScalarNull *Null = &Null{scalar{Type: arrow.Null, Valid: false}}
)

type PrimitiveScalar interface {
	Scalar
	Data() []byte
}

type Boolean struct {
	scalar
	Value bool
}

// by the time we get here we already know that the rhs is the right type
func (n *Boolean) equals(rhs Scalar) bool {
	return n.Value == rhs.(*Boolean).Value
}

func (s *Boolean) value() interface{} { return s.Value }

func (s *Boolean) Data() []byte {
	return (*[1]byte)(unsafe.Pointer(&s.Value))[:]
}

func (s *Boolean) String() string {
	if !s.Valid {
		return "null"
	}
	val, err := s.CastTo(arrow.BinaryTypes.String)
	if err != nil {
		return "..."
	}
	return string(val.(*String).Value.Bytes())
}

func (s *Boolean) CastTo(dt arrow.DataType) (Scalar, error) {
	if !s.Valid {
		return MakeNullScalar(dt), nil
	}

	val := 0
	if s.Value {
		val = 1
	}

	switch dt.ID() {
	case arrow.UINT8:
		return NewUint8Scalar(uint8(val)), nil
	case arrow.INT8:
		return NewInt8Scalar(int8(val)), nil
	case arrow.UINT16:
		return NewUint16Scalar(uint16(val)), nil
	case arrow.INT16:
		return NewInt16Scalar(int16(val)), nil
	case arrow.UINT32:
		return NewUint32Scalar(uint32(val)), nil
	case arrow.INT32:
		return NewInt32Scalar(int32(val)), nil
	case arrow.UINT64:
		return NewUint64Scalar(uint64(val)), nil
	case arrow.INT64:
		return NewInt64Scalar(int64(val)), nil
	case arrow.FLOAT16:
		return NewFloat16Scalar(float16.New(float32(val))), nil
	case arrow.FLOAT32:
		return NewFloat32Scalar(float32(val)), nil
	case arrow.FLOAT64:
		return NewFloat64Scalar(float64(val)), nil
	default:
		return nil, xerrors.Errorf("invalid scalar cast from type bool to type %s", dt)
	}
}

func NewBooleanScalar(val bool) *Boolean {
	return &Boolean{scalar{arrow.FixedWidthTypes.Boolean, true}, val}
}

type Float16 struct {
	scalar
	Value float16.Num
}

func (s *Float16) value() interface{} { return s.Value }

func (f *Float16) Data() []byte {
	return (*[arrow.Float16SizeBytes]byte)(unsafe.Pointer(&f.Value))[:]
}
func (f *Float16) equals(rhs Scalar) bool {
	return f.Value == rhs.(*Float16).Value
}
func (f *Float16) CastTo(to arrow.DataType) (Scalar, error) {
	if !f.Valid {
		return MakeNullScalar(to), nil
	}

	if r, ok := numericMap[to.ID()]; ok {
		return convertToNumeric(reflect.ValueOf(f.Value.Float32()), r.valueType, r.scalarFunc), nil
	}

	if to.ID() == arrow.BOOL {
		return NewBooleanScalar(f.Value.Uint16() != 0), nil
	} else if to.ID() == arrow.STRING {
		return NewStringScalar(f.Value.String()), nil
	}

	return nil, xerrors.Errorf("cannot cast non-null float16 scalar to type %s", to)
}

func (s *Float16) String() string {
	if !s.Valid {
		return "null"
	}
	val, err := s.CastTo(arrow.BinaryTypes.String)
	if err != nil {
		return "..."
	}
	return string(val.(*String).Value.Bytes())
}

func NewFloat16ScalarFromFloat32(val float32) *Float16 {
	return NewFloat16Scalar(float16.New(val))
}

func NewFloat16Scalar(val float16.Num) *Float16 {
	return &Float16{scalar{arrow.FixedWidthTypes.Float16, true}, val}
}

type Decimal128 struct {
	scalar
	Value decimal128.Num
}

func (s *Decimal128) value() interface{} { return s.Value }

func (s *Decimal128) String() string {
	if !s.Valid {
		return "null"
	}
	val, err := s.CastTo(arrow.BinaryTypes.String)
	if err != nil {
		return "..."
	}
	return string(val.(*String).Value.Bytes())
}

func (s *Decimal128) equals(rhs Scalar) bool {
	return s.Value == rhs.(*Decimal128).Value
}

func (s *Decimal128) CastTo(to arrow.DataType) (Scalar, error) {
	if !s.Valid {
		return MakeNullScalar(to), nil
	}

	switch to.ID() {
	case arrow.DECIMAL:
		return NewDecimal128Scalar(s.Value, to), nil
	case arrow.STRING:
		dt := s.Type.(*arrow.Decimal128Type)
		scale := big.NewFloat(math.Pow10(int(dt.Scale)))
		val := (&big.Float{}).SetInt(s.Value.BigInt())
		return NewStringScalar(val.Quo(val, scale).Text('g', int(dt.Precision))), nil
	}

	return nil, xerrors.Errorf("cannot cast non-nil decimal128 scalar to type %s", to)
}

func NewDecimal128Scalar(val decimal128.Num, typ arrow.DataType) *Decimal128 {
	return &Decimal128{scalar{typ, true}, val}
}

type Extension struct {
	scalar
	Value Scalar
}

func (s *Extension) value() interface{} { return s.Value }
func (s *Extension) equals(rhs Scalar) bool {
	return Equals(s.Value, rhs.(*Extension).Value)
}
func (e *Extension) Validate() (err error) {
	if err = e.scalar.Validate(); err != nil {
		return err
	}

	if !e.Valid {
		if e.Value != nil {
			err = xerrors.Errorf("null %s scalar has storage value", e.Type)
		}
		return
	}

	switch {
	case e.Value == nil:
		err = xerrors.Errorf("non-null %s scalar doesn't have a storage value", e.Type)
	case !e.Value.IsValid():
		err = xerrors.Errorf("non-null %s scalar has a null storage value", e.Type)
	default:
		if err = e.Value.Validate(); err != nil {
			err = xerrors.Errorf("%s scalar fails validation for storage value: %w", e.Type, err)
		}
	}
	return
}

func (e *Extension) ValidateFull() error {
	if err := e.Validate(); err != nil {
		return err
	}

	if e.Valid {
		return e.Value.ValidateFull()
	}
	return nil
}

func (s *Extension) CastTo(to arrow.DataType) (Scalar, error) {
	if !s.Valid {
		return MakeNullScalar(to), nil
	}

	if arrow.TypeEqual(s.Type, to) {
		return s, nil
	}

	return nil, xerrors.Errorf("cannot cast non-null extension scalar of type %s to type %s", s.Type, to)
}

func (s *Extension) String() string {
	if !s.Valid {
		return "null"
	}
	val, err := s.CastTo(arrow.BinaryTypes.String)
	if err != nil {
		return "..."
	}
	return string(val.(*String).Value.Bytes())
}

func NewExtensionScalar(storage Scalar, typ arrow.DataType) *Extension {
	return &Extension{scalar{typ, true}, storage}
}

func convertToNumeric(v reflect.Value, to reflect.Type, fn reflect.Value) Scalar {
	return fn.Call([]reflect.Value{v.Convert(to)})[0].Interface().(Scalar)
}

func MakeNullScalar(dt arrow.DataType) Scalar {
	return makeNullFn[byte(dt.ID()&0x1f)](dt)
}

func unsupportedScalarType(dt arrow.DataType) Scalar {
	panic("unsupported scalar data type: " + dt.ID().String())
}

func invalidScalarType(dt arrow.DataType) Scalar {
	panic("invalid scalar type: " + dt.ID().String())
}

type scalarMakeNullFn func(arrow.DataType) Scalar

var makeNullFn [32]scalarMakeNullFn

func init() {
	makeNullFn = [...]scalarMakeNullFn{
		arrow.NULL:              func(dt arrow.DataType) Scalar { return ScalarNull },
		arrow.BOOL:              func(dt arrow.DataType) Scalar { return &Boolean{scalar: scalar{dt, false}} },
		arrow.UINT8:             func(dt arrow.DataType) Scalar { return &Uint8{scalar: scalar{dt, false}} },
		arrow.INT8:              func(dt arrow.DataType) Scalar { return &Int8{scalar: scalar{dt, false}} },
		arrow.UINT16:            func(dt arrow.DataType) Scalar { return &Uint16{scalar: scalar{dt, false}} },
		arrow.INT16:             func(dt arrow.DataType) Scalar { return &Int16{scalar: scalar{dt, false}} },
		arrow.UINT32:            func(dt arrow.DataType) Scalar { return &Uint32{scalar: scalar{dt, false}} },
		arrow.INT32:             func(dt arrow.DataType) Scalar { return &Int32{scalar: scalar{dt, false}} },
		arrow.UINT64:            func(dt arrow.DataType) Scalar { return &Uint64{scalar: scalar{dt, false}} },
		arrow.INT64:             func(dt arrow.DataType) Scalar { return &Int64{scalar: scalar{dt, false}} },
		arrow.FLOAT16:           func(dt arrow.DataType) Scalar { return &Float16{scalar: scalar{dt, false}} },
		arrow.FLOAT32:           func(dt arrow.DataType) Scalar { return &Float32{scalar: scalar{dt, false}} },
		arrow.FLOAT64:           func(dt arrow.DataType) Scalar { return &Float64{scalar: scalar{dt, false}} },
		arrow.STRING:            func(dt arrow.DataType) Scalar { return &String{&Binary{scalar: scalar{dt, false}}} },
		arrow.BINARY:            func(dt arrow.DataType) Scalar { return &Binary{scalar: scalar{dt, false}} },
		arrow.FIXED_SIZE_BINARY: func(dt arrow.DataType) Scalar { return &FixedSizeBinary{&Binary{scalar: scalar{dt, false}}} },
		arrow.DATE32:            func(dt arrow.DataType) Scalar { return &Date32{scalar: scalar{dt, false}} },
		arrow.DATE64:            func(dt arrow.DataType) Scalar { return &Date64{scalar: scalar{dt, false}} },
		arrow.TIMESTAMP:         func(dt arrow.DataType) Scalar { return &Timestamp{scalar: scalar{dt, false}} },
		arrow.TIME32:            func(dt arrow.DataType) Scalar { return &Time32{scalar: scalar{dt, false}} },
		arrow.TIME64:            func(dt arrow.DataType) Scalar { return &Time64{scalar: scalar{dt, false}} },
		arrow.INTERVAL: func(dt arrow.DataType) Scalar {
			if arrow.TypeEqual(dt, arrow.FixedWidthTypes.MonthInterval) {
				return &MonthInterval{scalar: scalar{dt, false}}
			}
			return &DayTimeInterval{scalar: scalar{dt, false}}
		},
		arrow.DECIMAL:         func(dt arrow.DataType) Scalar { return &Decimal128{scalar: scalar{dt, false}} },
		arrow.LIST:            func(dt arrow.DataType) Scalar { return &List{scalar: scalar{dt, false}} },
		arrow.STRUCT:          func(dt arrow.DataType) Scalar { return &Struct{scalar: scalar{dt, false}} },
		arrow.UNION:           unsupportedScalarType,
		arrow.DICTIONARY:      unsupportedScalarType,
		arrow.MAP:             func(dt arrow.DataType) Scalar { return &Map{&List{scalar: scalar{dt, false}}} },
		arrow.EXTENSION:       func(dt arrow.DataType) Scalar { return &Extension{scalar: scalar{dt, false}} },
		arrow.FIXED_SIZE_LIST: func(dt arrow.DataType) Scalar { return &FixedSizeList{&List{scalar: scalar{dt, false}}} },
		arrow.DURATION:        func(dt arrow.DataType) Scalar { return &Duration{scalar: scalar{dt, false}} },

		// invalid data types to fill out array size 2⁵-1
		31: invalidScalarType,
	}

	f := numericMap[arrow.FLOAT16]
	f.scalarFunc = reflect.ValueOf(NewFloat16ScalarFromFloat32)
	f.valueType = reflect.TypeOf(float32(0))
	numericMap[arrow.FLOAT16] = f
}

func GetScalar(arr array.Interface, idx int) (Scalar, error) {
	switch arr := arr.(type) {
	case *array.Binary:
		buf := memory.NewBufferBytes(arr.Value(idx))
		defer buf.Release()
		return NewBinaryScalar(buf, arr.DataType()), nil
	case *array.Boolean:
		return NewBooleanScalar(arr.Value(idx)), nil
	case *array.Date32:
		return NewDate32Scalar(arr.Value(idx)), nil
	case *array.Date64:
		return NewDate64Scalar(arr.Value(idx)), nil
	case *array.DayTimeInterval:
		return NewDayTimeIntervalScalar(arr.Value(idx)), nil
	case *array.Decimal128:
		return NewDecimal128Scalar(arr.Value(idx), arr.DataType()), nil
	case *array.Duration:
		return NewDurationScalar(arr.Value(idx), arr.DataType()), nil
	case array.ExtensionArray:
		storage, err := GetScalar(arr.Storage(), idx)
		if err != nil {
			return nil, err
		}
		return NewExtensionScalar(storage, arr.DataType()), nil
	case *array.FixedSizeBinary:
		buf := memory.NewBufferBytes(arr.Value(idx))
		defer buf.Release()
		return NewFixedSizeBinaryScalar(buf, arr.DataType()), nil
	case *array.FixedSizeList:
		size := int(arr.DataType().(*arrow.FixedSizeListType).Len())
		return NewFixedSizeListScalarWithType(array.NewSlice(arr.ListValues(), int64(idx*size), int64((idx+1)*size)), arr.DataType()), nil
	case *array.Float16:
		return NewFloat16Scalar(arr.Value(idx)), nil
	case *array.Float32:
		return NewFloat32Scalar(arr.Value(idx)), nil
	case *array.Float64:
		return NewFloat64Scalar(arr.Value(idx)), nil
	case *array.Int8:
		return NewInt8Scalar(arr.Value(idx)), nil
	case *array.Int16:
		return NewInt16Scalar(arr.Value(idx)), nil
	case *array.Int32:
		return NewInt32Scalar(arr.Value(idx)), nil
	case *array.Int64:
		return NewInt64Scalar(arr.Value(idx)), nil
	case *array.Uint8:
		return NewUint8Scalar(arr.Value(idx)), nil
	case *array.Uint16:
		return NewUint16Scalar(arr.Value(idx)), nil
	case *array.Uint32:
		return NewUint32Scalar(arr.Value(idx)), nil
	case *array.Uint64:
		return NewUint64Scalar(arr.Value(idx)), nil
	case *array.List:
		offsets := arr.Offsets()
		return NewListScalar(array.NewSlice(arr.ListValues(), int64(offsets[idx]), int64(offsets[idx+1]))), nil
	case *array.Map:
		offsets := arr.Offsets()
		return NewMapScalar(array.NewSlice(arr.ListValues(), int64(offsets[idx]), int64(offsets[idx+1]))), nil
	case *array.MonthInterval:
		return NewMonthIntervalScalar(arr.Value(idx)), nil
	case *array.Null:
		return ScalarNull, nil
	case *array.String:
		return NewStringScalar(arr.Value(idx)), nil
	case *array.Struct:
		children := make(Vector, arr.NumField())
		for i := range children {
			child, err := GetScalar(arr.Field(i), idx)
			if err != nil {
				return nil, err
			}
			children[i] = child
		}
		return NewStructScalar(children, arr.DataType()), nil
	case *array.Time32:
		return NewTime32Scalar(arr.Value(idx), arr.DataType()), nil
	case *array.Time64:
		return NewTime64Scalar(arr.Value(idx), arr.DataType()), nil
	case *array.Timestamp:
		return NewTimestampScalar(arr.Value(idx), arr.DataType()), nil
	}

	return nil, xerrors.Errorf("cannot create scalar from array of type %s", arr.DataType())
}
