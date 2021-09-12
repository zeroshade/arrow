#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -ex

# this is separate from the regular go_test.sh cdata tests in that these
# tests actually require libarrow to be available on the system and
# accessible via pkg-config

source_dir=${1}/go
pushd ${source_dir}/arrow

case "$(uname)" in
    MINGW*)
        # TODO(mtopol): add flags to CGO_LDFLAGS and CGO_CPPFLAGS for cgo to link
        # against arrow.dll on windows
        echo "unless -larrow linker flag can find the arrow.dll, this will fail"
        ;;
esac 

go test -v -tags ccalloc ./memory
