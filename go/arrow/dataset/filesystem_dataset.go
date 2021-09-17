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
	"io/fs"
	"path/filepath"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/compute"
)

type FileSource struct {
	fs      fs.FS
	dirpath string
	info    fs.FileInfo
}

func (src *FileSource) Path() string { return filepath.Join(src.dirpath, src.info.Name()) }

type FileFormat interface {
	TypeName() string
	Equals(FileFormat) bool
	IsSupported(*FileSource) (bool, error)
	Inspect(*FileSource) (*arrow.Schema, error)
	ScanFile(opts *ScanOptions, file *FileFragment) (ScanTaskIterator, error)
	ScanBatches(opts *ScanOptions, file *FileFragment) (RecordGenerator, error)
	MakeFragment(*FileSource) (*FileFragment, error)
}

type FileFragment struct {
	source    *FileSource
	format    FileFormat
	partition compute.Expression
}

func (ff *FileFragment) Format() FileFormat { return ff.format }
func (ff *FileFragment) String() string     { return ff.source.Path() }
func (ff *FileFragment) ReadPhysicalSchema() (*arrow.Schema, error) {
	return ff.format.Inspect(ff.source)
}
func (ff *FileFragment) Scan(opts *ScanOptions) (ScanTaskIterator, error) {
	return ff.format.ScanFile(opts, ff)
}
func (ff *FileFragment) TypeName() string                  { return ff.format.TypeName() }
func (ff *FileFragment) PartitionExpr() compute.Expression { return ff.partition }
