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

// +build cgo

#if defined(_WIN32) || defined(__CYGWIN__)
#define ARROW_STATIC 1
#endif 

#include <iostream>
#include "abi.h"
#include "../memory/internal/cgoarrow/helpers.h"
#include "arrow/record_batch.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/compute/exec.h"

ExecContext arrow_compute_default_context() {
    std::shared_ptr<arrow::compute::ExecContext> ctx(arrow::compute::default_exec_context(), [](arrow::compute::ExecContext*){});
    return create_ref<arrow::compute::ExecContext>(ctx);
}

void arrow_compute_release_context(ExecContext ctx) { release_ref<arrow::compute::ExecContext>(ctx); }

int64_t arrow_compute_get_exec_chunksize(ExecContext ctx) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    return exec_context->exec_chunksize();
}

void arrow_compute_set_exec_chunksize(ExecContext ctx, int64_t chunksize) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    return exec_context->set_exec_chunksize(chunksize);
}

int arrow_compute_execute_scalar_expr(ExecContext ctx,
                                  struct ArrowComputeInputOutput* partial_input,
                                  const uint8_t* serialized_expr, const int serialized_len,
                                  struct ArrowComputeInputOutput* result) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);    
    auto expr_result = arrow::compute::Deserialize(std::make_shared<arrow::Buffer>(serialized_expr, serialized_len));

    if (!expr_result.ok()) {
        std::cerr << "msg: " << expr_result.status().message() << std::endl;
        return 1;
    }

    auto input = arrow::ImportRecordBatch(partial_input->data, partial_input->schema);
    if (!input.ok()) {
        std::cerr << input.status().message() << std::endl;
        return 2;
    }        
    auto batch = input.MoveValueUnsafe();

    auto exec_batch = arrow::compute::ExecBatch(*batch);
    const auto& expr = expr_result.ValueOrDie().Bind(*batch->schema(), exec_context.get());
    auto output = arrow::compute::ExecuteScalarExpression(expr.ValueOrDie(), exec_batch, exec_context.get());        

    if (!output.ok()) {
        return 3;
    }

    const auto& datum = output.ValueOrDie();
    auto status = arrow::ExportArray(*datum.make_array(), result->data, result->schema);
    if (!status.ok()) {
        return 4;
    }
    
    return 0;
}
