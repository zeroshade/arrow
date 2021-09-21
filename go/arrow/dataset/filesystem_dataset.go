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
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/compute"
)

type FileSource struct {
	fs      fs.StatFS
	dirpath string
	info    fs.FileInfo
}

func (src *FileSource) Open() (fs.File, error) {
	return src.fs.Open(filepath.Join(src.dirpath, src.info.Name()))
}
func (src *FileSource) Path() string { return filepath.Join(src.dirpath, src.info.Name()) }

func NewFileSource(path string, filesystem fs.StatFS) (*FileSource, error) {
	info, err := fs.Stat(filesystem, path)
	if err != nil {
		return nil, err
	}

	return &FileSource{
		fs:      filesystem,
		dirpath: filepath.Dir(path),
		info:    info,
	}, nil
}

type fragCfg struct {
	partition      compute.Expression
	physicalSchema *arrow.Schema
}

type MakeFragmentOption func(*fragCfg)

func WithPartition(expr compute.Expression) MakeFragmentOption {
	return func(c *fragCfg) {
		c.partition = expr
	}
}

func WithPhysicalSchema(s *arrow.Schema) MakeFragmentOption {
	return func(c *fragCfg) {
		c.physicalSchema = s
	}
}

type FileFormat interface {
	TypeName() string
	Equals(FileFormat) bool
	IsSupported(*FileSource) (bool, error)
	Inspect(*FileSource) (*arrow.Schema, error)
	ScanFile(opts *ScanOptions, file *FileFragment) (ScanTaskIterator, error)
	// ScanBatches(opts *ScanOptions, file *FileFragment) (RecordGenerator, error)
	MakeFragment(*FileSource, ...MakeFragmentOption) (*FileFragment, error)
}

type FileFragment struct {
	source    *FileSource
	format    FileFormat
	partition compute.Expression
}

func (ff *FileFragment) Source() *FileSource { return ff.source }
func (ff *FileFragment) Format() FileFormat  { return ff.format }
func (ff *FileFragment) String() string      { return ff.source.Path() }
func (ff *FileFragment) ReadPhysicalSchema() (*arrow.Schema, error) {
	return ff.format.Inspect(ff.source)
}
func (ff *FileFragment) Scan(opts *ScanOptions) (ScanTaskIterator, error) {
	return ff.format.ScanFile(opts, ff)
}
func (ff *FileFragment) TypeName() string                  { return ff.format.TypeName() }
func (ff *FileFragment) PartitionExpr() compute.Expression { return ff.partition }

func NewFileFragment(source *FileSource, format FileFormat, partition compute.Expression) *FileFragment {
	return &FileFragment{source, format, partition}
}

func NewFileFragmentWithOptions(source *FileSource, format FileFormat, opts ...MakeFragmentOption) *FileFragment {
	cfg := fragCfg{partition: compute.NewLiteral(true)}
	for _, o := range opts {
		o(&cfg)
	}

	return &FileFragment{source, format, cfg.partition}
}

type fragmentSubtrees struct {
	forest           compute.Forest
	fragsAndSubtrees []interface{}
}

type FileSystemDatasetOptions struct {
	ExcludeInvalid      bool
	Schema              *arrow.Schema
	InspectOpts         InspectOptions
	ValidateFragments   bool
	Partitioning        Partitioning
	PartitioningFactory PartitioningFactory
	PartitionBaseDir    string
}

func (fopt *FileSystemDatasetOptions) getOrInferSchema(paths []string) (*arrow.Schema, error) {
	if fopt.Partitioning == nil && fopt.PartitioningFactory == nil {
		fopt.Partitioning = DefaultPartitioning{}
	}

	if fopt.Partitioning != nil {
		return fopt.Partitioning.Schema(), nil
	}
	return fopt.PartitioningFactory.Inspect(paths)
}

type fsDatasetCfg struct {
	files         []*FileSource
	fs            fs.StatFS
	format        FileFormat
	options       *FileSystemDatasetOptions
	rootPartition compute.Expression
}

type InspectOptions struct {
	Fragments int
}

func (cfg *fsDatasetCfg) inspectSchemas(opts InspectOptions) ([]*arrow.Schema, error) {
	schemas := make([]*arrow.Schema, 0, len(cfg.files))
	hasFragmentLimit := opts.Fragments >= 0
	fragments := opts.Fragments
	for _, f := range cfg.files {
		fragments--
		if hasFragmentLimit && fragments < 0 {
			break
		}

		s, err := cfg.format.Inspect(f)
		if err != nil {
			return nil, fmt.Errorf("error creating dataset. could not read schema from '%s': %w. is this a '%s' file?", f.Path(), err, cfg.format.TypeName())
		}

		schemas = append(schemas, s)
	}

	partitionSchema, err := cfg.options.getOrInferSchema(stripPrefixAndFilenames(cfg.files, cfg.options.PartitionBaseDir))
	if err != nil {
		return nil, err
	}
	schemas = append(schemas, partitionSchema)
	return schemas, nil
}

func (cfg *fsDatasetCfg) inspect() (*arrow.Schema, error) {
	schemas, err := cfg.inspectSchemas(cfg.options.InspectOpts)
	if err != nil {
		return nil, err
	}

	if len(schemas) == 0 {
		return arrow.NewSchema([]arrow.Field{}, nil), nil
	}

	return arrow.UnifySchemas(schemas)
}

func (cfg *fsDatasetCfg) makeDataset() (*FileSystemDataset, error) {
	schema := cfg.options.Schema
	missingSchema := schema == nil
	var err error
	if missingSchema {
		schema, err = cfg.inspect()
		if err != nil {
			return nil, err
		}
	}

	if cfg.options.ValidateFragments && !missingSchema {
		schemas, err := cfg.inspectSchemas(cfg.options.InspectOpts)
		if err != nil {
			return nil, err
		}

		for _, s := range schemas {
			if err := arrow.SchemasAreCompatible([]*arrow.Schema{schema, s}, arrow.ConflictMerge); err != nil {
				return nil, err
			}
		}
	}

	partitioning := cfg.options.Partitioning
	if partitioning == nil {
		partitioning, err = cfg.options.PartitioningFactory.Make(schema)
		if err != nil {
			return nil, err
		}
	}

	fragments := make([]*FileFragment, len(cfg.files))
	for i, f := range cfg.files {
		fixed := stripPrefixFilename(f.Path(), cfg.options.PartitionBaseDir)
		partition, err := partitioning.Parse(fixed)
		if err != nil {
			return nil, err
		}
		fragments[i], err = cfg.format.MakeFragment(f, WithPartition(partition))
		if err != nil {
			return nil, err
		}
	}

	return NewFileSystemDataset(schema, cfg.rootPartition, cfg.format, cfg.fs, fragments, partitioning), nil
}

func NewFileSystemDatasetFromPaths(paths []string, filesystem fs.StatFS, format FileFormat, options *FileSystemDatasetOptions) (*FileSystemDataset, error) {
	cfg := &fsDatasetCfg{
		files:         make([]*FileSource, 0, len(paths)),
		fs:            filesystem,
		format:        format,
		options:       options,
		rootPartition: compute.NewLiteral(true),
	}
	for _, p := range paths {
		source, err := NewFileSource(p, filesystem)
		if err != nil {
			return nil, err
		}
		if options.ExcludeInvalid {
			supported, err := format.IsSupported(source)
			if err != nil {
				return nil, err
			}
			if !supported {
				continue
			}
		}
		cfg.files = append(cfg.files, source)
	}
	return cfg.makeDataset()
}

type FileSystemDataset struct {
	dataset

	format       FileFormat
	fs           fs.StatFS
	fragments    []*FileFragment
	partitioning Partitioning
	subtrees     fragmentSubtrees
}

func NewFileSystemDataset(schema *arrow.Schema, root compute.Expression, format FileFormat, filesystem fs.StatFS, fragments []*FileFragment, partitioning Partitioning) *FileSystemDataset {
	out := &FileSystemDataset{
		dataset:      dataset{schema: schema, partition: root},
		format:       format,
		fs:           filesystem,
		fragments:    fragments,
		partitioning: partitioning,
	}

	out.setupSubtreePruning()
	return out
}

func (fsi *FileSystemDataset) setupSubtreePruning() {
	var impl compute.Subtree
	encoded := compute.EncodedList(impl.EncodeGuarantees(func(i int) compute.Expression { return fsi.fragments[i].PartitionExpr() }, len(fsi.fragments)))
	sort.Sort(compute.ByGuarantee{EncodedList: encoded})
	for _, e := range encoded {
		if e.Index != nil {
			fsi.subtrees.fragsAndSubtrees = append(fsi.subtrees.fragsAndSubtrees, *e.Index)
		} else {
			fsi.subtrees.fragsAndSubtrees = append(fsi.subtrees.fragsAndSubtrees, impl.GetSubtreeExpr(e))
		}
	}

	fsi.subtrees.forest = compute.NewForest(len(encoded), compute.IsAncestor(encoded))
}

func makeFileFragmentItr(fragments []*FileFragment) FragmentIterator {
	itr := make(chan FragmentMessage)
	go func() {
		defer close(itr)
		for _, f := range fragments {
			itr <- FragmentMessage{Fragment: f}
		}
	}()
	return itr
}

func (fsi *FileSystemDataset) GetFragments() (FragmentIterator, error) {
	return fsi.GetFragmentsCond(compute.NewLiteral(true))
}

func (fsi *FileSystemDataset) GetFragmentsCond(predicate compute.Expression) (FragmentIterator, error) {
	if predicate.Equals(compute.NewLiteral(true)) {
		// trivial predicate, skip subtree pruning
		return makeFileFragmentItr(fsi.fragments), nil
	}

	indices := make([]int, 0)
	predicates := []compute.Expression{predicate}
	err := fsi.subtrees.forest.Visit(func(r compute.Ref) (bool, error) {
		v := fsi.subtrees.fragsAndSubtrees[r.Idx]

		if fragIndex, ok := v.(int); ok {
			indices = append(indices, fragIndex)
			return false, nil
		}

		subtreeExpr := v.(compute.Expression)
		simplified, err := compute.SimplifyWithGuarantee(predicates[len(predicates)-1], subtreeExpr)
		if err != nil {
			return false, err
		}

		if !simplified.IsSatisfiable() {
			return false, nil
		}

		predicates = append(predicates, simplified)
		return true, nil
	}, func(r compute.Ref) {
		predicates = predicates[:len(predicates)-1]
	})
	if err != nil {
		return nil, err
	}

	sort.Ints(indices)
	fragments := make([]*FileFragment, len(indices))
	for i, f := range indices {
		fragments[i] = fsi.fragments[f]
	}
	return makeFileFragmentItr(fragments), nil
}

func (fsi *FileSystemDataset) TypeName() string { return "filesystem" }

func (fsi *FileSystemDataset) ReplaceSchema(schema *arrow.Schema) (Dataset, error) {
	if err := checkProjectable(fsi.schema, schema); err != nil {
		return nil, err
	}

	return NewFileSystemDataset(schema, fsi.partition, fsi.format, fsi.fs, fsi.fragments, nil), nil
}
