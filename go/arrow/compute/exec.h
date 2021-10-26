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

#include <stdbool.h>
#include <stdint.h>

#include "arrow/c/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// default chunk size for execution
extern const int64_t kDefaultExecChunk;
// values for the ValueDescr shape enums
extern const int arrow_shape_any;
extern const int arrow_shape_array;
extern const int arrow_shape_scalar;
// check the function registry and report whether the named function
// is a scalar function or not.
bool arrow_compute_function_is_scalar(const char* funcname);

typedef uintptr_t ExecContext;
// return a reference to the default execution context
ExecContext arrow_compute_default_context();
void arrow_compute_release_context(ExecContext ctx);
// get the current chunk size from an execution context
int64_t arrow_compute_get_exec_chunksize(ExecContext ctx);
// set the chunksize for an execution context
void arrow_compute_set_exec_chunksize(ExecContext ctx, const int64_t chunksize);

// pointer to a holder of the bound expression with it's C++ information
typedef uintptr_t BoundExpression;

struct arrow_postbind_info {
  // reference to the bound expression
  BoundExpression bound;
  // serialized new expression (simplification may have happened)
  const uint8_t* serialized_data;
  int64_t serialized_len;
  // if not null, contains the error message from binding attempt
  const char* status;
  // shape matching one of the enum values of arrow_shape_any, array or scalar
  int shape;
  // the valuedescr type of the bound expression
  struct ArrowSchema* type;
  // the field index of the bound expression if relevant
  int index;
};

// bind an expression to the given input schema type and return the resolved
// field type simplifications and value description. Can pass 0/null for the
// ctx for it to use the default execution context.
struct arrow_postbind_info arrow_compute_bind_expr(ExecContext ctx,
                                                   struct ArrowSchema* schema,
                                                   const uint8_t* serialized_expr,
                                                   const int serialized_len);

// retrieve the last error status associated with the passed in expression
const char* arrow_compute_expr_last_status(BoundExpression bound);

// retrieve the datatype of the bound expression. returns true if successful.
// if false is returned, arrow_compute_expr_last_status can be called to
// retrieve the error message.
bool arrow_compute_bound_expr_type(BoundExpression bound, struct ArrowSchema* out);
// retrieve the binding information of a specific argument to a bound Call expression
struct arrow_postbind_info arrow_compute_get_bound_arg(BoundExpression bound, size_t idx);
// release the C memory allocated for a bound expression.
void arrow_compute_bound_expr_release(BoundExpression bound);

// represent a Datum for passing between Go and C
struct ArrowDatum {
  // should match one of static_cast<int>(DatumKind)
  int datum_type;
  struct ArrowSchema* schema;
  // to handle chunked arrays and tables we pass an array of arrays
  // otherwise this will be 1.
  int num_data;
  struct ArrowArray** data;
};

// execute a scalar expression using the given schema and input record batch or array
struct ArrowDatum* arrow_compute_exec_scalar_expr(BoundExpression bound,
                                                  struct ArrowSchema* full_schema,
                                                  struct ArrowDatum* input,
                                                  ExecContext ctx);
// call a function directly by providing an execution context (or null), the function
// to call, a list of arguments and serialized options.
struct ArrowDatum* arrow_compute_call_function(
    ExecContext ctx, const char* func_name, struct ArrowDatum** args, const int num_args,
    const char* options_type, const uint8_t* serialized_opts, const int serialized_len);

#ifdef __cplusplus
}
#endif
