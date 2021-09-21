// Code generated by datatype_numeric.gen.go.tmpl. DO NOT EDIT.

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

type Int8Type struct{}

func (t *Int8Type) ID() Type            { return INT8 }
func (t *Int8Type) Name() string        { return "int8" }
func (t *Int8Type) String() string      { return "int8" }
func (t *Int8Type) BitWidth() int       { return 8 }
func (t *Int8Type) Fingerprint() string { return typeIdFingerprint(t) }

type Int16Type struct{}

func (t *Int16Type) ID() Type            { return INT16 }
func (t *Int16Type) Name() string        { return "int16" }
func (t *Int16Type) String() string      { return "int16" }
func (t *Int16Type) BitWidth() int       { return 16 }
func (t *Int16Type) Fingerprint() string { return typeIdFingerprint(t) }

type Int32Type struct{}

func (t *Int32Type) ID() Type            { return INT32 }
func (t *Int32Type) Name() string        { return "int32" }
func (t *Int32Type) String() string      { return "int32" }
func (t *Int32Type) BitWidth() int       { return 32 }
func (t *Int32Type) Fingerprint() string { return typeIdFingerprint(t) }

type Int64Type struct{}

func (t *Int64Type) ID() Type            { return INT64 }
func (t *Int64Type) Name() string        { return "int64" }
func (t *Int64Type) String() string      { return "int64" }
func (t *Int64Type) BitWidth() int       { return 64 }
func (t *Int64Type) Fingerprint() string { return typeIdFingerprint(t) }

type Uint8Type struct{}

func (t *Uint8Type) ID() Type            { return UINT8 }
func (t *Uint8Type) Name() string        { return "uint8" }
func (t *Uint8Type) String() string      { return "uint8" }
func (t *Uint8Type) BitWidth() int       { return 8 }
func (t *Uint8Type) Fingerprint() string { return typeIdFingerprint(t) }

type Uint16Type struct{}

func (t *Uint16Type) ID() Type            { return UINT16 }
func (t *Uint16Type) Name() string        { return "uint16" }
func (t *Uint16Type) String() string      { return "uint16" }
func (t *Uint16Type) BitWidth() int       { return 16 }
func (t *Uint16Type) Fingerprint() string { return typeIdFingerprint(t) }

type Uint32Type struct{}

func (t *Uint32Type) ID() Type            { return UINT32 }
func (t *Uint32Type) Name() string        { return "uint32" }
func (t *Uint32Type) String() string      { return "uint32" }
func (t *Uint32Type) BitWidth() int       { return 32 }
func (t *Uint32Type) Fingerprint() string { return typeIdFingerprint(t) }

type Uint64Type struct{}

func (t *Uint64Type) ID() Type            { return UINT64 }
func (t *Uint64Type) Name() string        { return "uint64" }
func (t *Uint64Type) String() string      { return "uint64" }
func (t *Uint64Type) BitWidth() int       { return 64 }
func (t *Uint64Type) Fingerprint() string { return typeIdFingerprint(t) }

type Float32Type struct{}

func (t *Float32Type) ID() Type            { return FLOAT32 }
func (t *Float32Type) Name() string        { return "float32" }
func (t *Float32Type) String() string      { return "float32" }
func (t *Float32Type) BitWidth() int       { return 32 }
func (t *Float32Type) Fingerprint() string { return typeIdFingerprint(t) }

type Float64Type struct{}

func (t *Float64Type) ID() Type            { return FLOAT64 }
func (t *Float64Type) Name() string        { return "float64" }
func (t *Float64Type) String() string      { return "float64" }
func (t *Float64Type) BitWidth() int       { return 64 }
func (t *Float64Type) Fingerprint() string { return typeIdFingerprint(t) }

type Date32Type struct{}

func (t *Date32Type) ID() Type            { return DATE32 }
func (t *Date32Type) Name() string        { return "date32" }
func (t *Date32Type) String() string      { return "date32" }
func (t *Date32Type) BitWidth() int       { return 32 }
func (t *Date32Type) Fingerprint() string { return typeIdFingerprint(t) }

type Date64Type struct{}

func (t *Date64Type) ID() Type            { return DATE64 }
func (t *Date64Type) Name() string        { return "date64" }
func (t *Date64Type) String() string      { return "date64" }
func (t *Date64Type) BitWidth() int       { return 64 }
func (t *Date64Type) Fingerprint() string { return typeIdFingerprint(t) }

var (
	PrimitiveTypes = struct {
		Int8    DataType
		Int16   DataType
		Int32   DataType
		Int64   DataType
		Uint8   DataType
		Uint16  DataType
		Uint32  DataType
		Uint64  DataType
		Float32 DataType
		Float64 DataType
		Date32  DataType
		Date64  DataType
	}{

		Int8:    &Int8Type{},
		Int16:   &Int16Type{},
		Int32:   &Int32Type{},
		Int64:   &Int64Type{},
		Uint8:   &Uint8Type{},
		Uint16:  &Uint16Type{},
		Uint32:  &Uint32Type{},
		Uint64:  &Uint64Type{},
		Float32: &Float32Type{},
		Float64: &Float64Type{},
		Date32:  &Date32Type{},
		Date64:  &Date64Type{},
	}
)
