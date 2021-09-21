// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "arrow/c/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t ExecContext;

struct ArrowComputeInputOutput {
    struct ArrowSchema* schema;
    struct ArrowArray* data;    
};

inline struct ArrowComputeInputOutput get_io() {
    return (struct ArrowComputeInputOutput) {
        .schema = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema)),
		.data = (struct ArrowArray*)malloc(sizeof(struct ArrowArray)),        
    };
}

inline void release_io(struct ArrowComputeInputOutput* io) {
  free(io->data);
  io->data = NULL;
  free(io->schema);
  io->schema = NULL;
}

ExecContext arrow_compute_default_context();
void arrow_compute_release_context(ExecContext ctx);
int64_t arrow_compute_get_exec_chunksize(ExecContext ctx);
void arrow_compute_set_exec_chunksize(ExecContext ctx, int64_t chunksize);

typedef uintptr_t BoundExpression;

BoundExpression arrow_compute_bind_expr(ExecContext ctx, struct ArrowSchema* schema,
    const uint8_t* serialized_expr, const int serialized_len);

void arrow_compute_bound_expr_type(BoundExpression bound, struct ArrowSchema* out);
void arrow_compute_bound_expr_release(BoundExpression bound);
uint64_t arrow_compute_bound_expr_hash(BoundExpression bound);
int arrow_compute_bound_expr_simplify_guarantee(BoundExpression expr, 
    const uint8_t* serialized_guarantee, const int serialized_len,
    BoundExpression* out);
bool arrow_compute_bound_is_satisfiable(BoundExpression bound);
bool arrow_compute_bound_is_scalar(BoundExpression bound);
bool arrow_compute_function_scalar(const char* funcname);
BoundExpression arrow_compute_get_bound_arg(BoundExpression bound, size_t idx);

int arrow_compute_execute_scalar_expr_schema(ExecContext ctx,
    struct ArrowSchema* full_schema, struct ArrowComputeInputOutput* partial_input,
    BoundExpression expr, struct ArrowComputeInputOutput* result);

int arrow_compute_execute_scalar_expr(ExecContext ctx, 
    struct ArrowComputeInputOutput* partial_input,
    const uint8_t* serialized_expr, const int serialized_len,
    struct ArrowComputeInputOutput* result);

int call_function(ExecContext ctx, const char* func_name, 
                  struct ArrowComputeInputOutput* args,
                  struct ArrowComputeInputOutput* options, 
                  struct ArrowComputeInputOutput* result);

#ifdef __cplusplus
}
#endif
