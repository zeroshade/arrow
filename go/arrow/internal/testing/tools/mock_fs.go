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

package tools

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

type mockFile struct {
	info *mockFileInfo

	rdr *bytes.Reader
}

func (m *mockFile) Stat() (fs.FileInfo, error) {
	return m.info, nil
}

func (m *mockFile) Read(p []byte) (int, error) {
	return m.rdr.Read(p)
}

func (m *mockFile) Close() error {
	m.rdr = nil
	return nil
}

type mockFileInfo struct {
	name     string
	contents []byte
	size     int64
	mode     fs.FileMode
	modTime  time.Time
	dir      bool
	sys      interface{}
}

func (m *mockFileInfo) Name() string       { return m.name }
func (m *mockFileInfo) Size() int64        { return m.size }
func (m *mockFileInfo) Mode() fs.FileMode  { return m.mode }
func (m *mockFileInfo) ModTime() time.Time { return m.modTime }
func (m *mockFileInfo) IsDir() bool        { return m.dir }
func (m *mockFileInfo) Sys() interface{}   { return m.sys }

type mockDirEntry struct {
	info mockFileInfo
	err  error

	children []*mockDirEntry
}

func (m *mockDirEntry) Name() string               { return m.info.name }
func (m *mockDirEntry) IsDir() bool                { return m.info.dir }
func (m *mockDirEntry) Type() fs.FileMode          { return m.info.mode }
func (m *mockDirEntry) Info() (fs.FileInfo, error) { return &m.info, m.err }

type MockFS struct {
	root *mockDirEntry
}

func (m *MockFS) Open(name string) (fs.File, error) {
	entry, _, err := m.findEntry(name)
	if err != nil {
		return nil, err
	}

	return &mockFile{
		info: &entry.info,
		rdr:  bytes.NewReader(entry.info.contents),
	}, nil
}

func (m *MockFS) Stat(name string) (fs.FileInfo, error) {
	entry, _, err := m.findEntry(name)
	if err != nil {
		return nil, err
	}

	return &entry.info, nil
}

func getSplitPath(path string) []string {
	out := make([]string, 0)
	for len(path) > 0 && path != "." && path != string(filepath.Separator) {
		out = append(out, filepath.Base(path))
		path = filepath.Dir(path)
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func (m *MockFS) findEntry(path string) (*mockDirEntry, []string, error) {
	plist := getSplitPath(path)

	if m.root == nil {
		m.root = &mockDirEntry{children: make([]*mockDirEntry, 0)}
	}

	root := m.root
	for len(plist) > 0 {
		var found *mockDirEntry
		for _, n := range root.children {
			if n.info.name == plist[0] {
				found = n
				break
			}
		}
		if found == nil {
			return root, plist, os.ErrNotExist
		}
		root = found
		plist = plist[1:]
	}
	return root, plist, nil
}

func (m *MockFS) CreateFile(path string, contents []byte) error {
	entry, remain, err := m.findEntry(path)
	if errors.Is(err, os.ErrNotExist) {
		for i, p := range remain {
			if entry.children == nil {
				entry.children = make([]*mockDirEntry, 0, 1)
			}

			next := &mockDirEntry{
				info: mockFileInfo{
					name: p,
					dir:  i != len(remain)-1,
					mode: fs.ModeDir,
					sys:  m,
				},
				children: make([]*mockDirEntry, 0),
			}
			entry.children = append(entry.children, next)
			entry = next
		}

		// entry is now the file itself
		entry.info.mode = fs.ModePerm
		entry.info.contents = contents
		return nil
	}

	if err != nil {
		return err
	}

	entry.info.contents = contents
	return nil
}
