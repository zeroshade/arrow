# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

project(
    'arrow',
    'cpp',
    'c',
    version: '20.0.0-SNAPSHOT',
    license: 'Apache-2.0',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c99',
        'warning_level=2',
        'cpp_std=c++17',
    ],
)

project_args = [
    '-Wno-unused-parameter',
    '-Wno-array-bounds',
    '-Wno-stringop-overflow',
    '-Wno-aggressive-loop-optimizations',
    '-Wno-nonnull',
]

c_compiler = meson.get_compiler('c')
c_args = c_compiler.get_supported_arguments(project_args)
add_project_arguments(c_args, language: 'c')

cpp_compiler = meson.get_compiler('cpp')
cpp_args = cpp_compiler.get_supported_arguments(project_args)
add_project_arguments(cpp_args, language: 'cpp')

git_id = get_option('git_id')
if git_id == ''
    git_id = run_command('git', 'log', '-n1', '--format=%H', check: false).stdout().strip()
endif

git_description = get_option('git_description')
if git_description == ''
    git_description = run_command('git', 'describe', '--tags', check: false).stdout().strip()
endif

needs_benchmarks = get_option('benchmarks').enabled()
needs_csv = get_option('csv').enabled()
needs_azure = get_option('azure').enabled()
needs_gcs = get_option('gcs').enabled()
needs_hdfs = get_option('hdfs').enabled()
needs_s3 = get_option('s3').enabled()
needs_filesystem = get_option('filesystem').enabled() or needs_azure or needs_gcs or needs_hdfs or needs_s3
needs_integration = get_option('integration').enabled()
needs_tests = get_option('tests').enabled()
needs_ipc = get_option('ipc').enabled() or needs_tests or needs_benchmarks
needs_testing = get_option('testing').enabled() or needs_tests or needs_benchmarks or needs_integration
needs_json = get_option('json').enabled() or needs_testing

subdir('src/arrow')
