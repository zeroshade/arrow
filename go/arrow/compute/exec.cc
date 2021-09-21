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
#include "arrow/array/util.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/compute/function_internal.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/registry.h"
#include "arrow/compute/function.h"

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

BoundExpression arrow_compute_bind_expr(ExecContext ctx, struct ArrowSchema* schema,
                            const uint8_t* serialized_expr, const int serialized_len) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    auto expr = arrow::compute::Deserialize(std::make_shared<arrow::Buffer>(serialized_expr, serialized_len));

    if (!expr.ok()) {
        std::cerr << "msg: " << expr.status().message() << std::endl;
        return 0;
    }

    auto schema_res = arrow::ImportSchema(schema);
    if (!schema_res.ok()) {
        std::cerr << "msg: " << schema_res.status().message() << std::endl;
        return 0;
    }

    auto expr_result = expr.ValueUnsafe().Bind(*schema_res.ValueUnsafe(), exec_context.get());
    if (!expr_result.ok()) {
        std::cerr << "msg: " << expr.status().message() << std::endl;
        return 0;
    }

    std::shared_ptr<arrow::compute::Expression> bound_expr = std::make_shared<arrow::compute::Expression>(expr_result.MoveValueUnsafe());
    return create_ref(bound_expr);
}

void arrow_compute_bound_expr_type(BoundExpression bound, struct ArrowSchema* out) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    arrow::ExportType(*(expr->type()), out);
}

void arrow_compute_bound_expr_release(BoundExpression bound) {
    release_ref<arrow::compute::Expression>(bound);
}

uint64_t arrow_compute_bound_expr_hash(BoundExpression bound) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    return expr->hash();
}

const char* arrow_compute_bound_expr_funcname(BoundExpression bound) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    auto* call = expr->call();
    if (!call) {
        return nullptr;
    }
    return call->function_name.c_str();
}

bool arrow_compute_bound_is_scalar(BoundExpression bound) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    return expr->IsScalarExpression();
}

bool arrow_compute_function_scalar(const char* funcname) {
    if (auto function = arrow::compute::GetFunctionRegistry()->GetFunction(funcname).ValueOr(nullptr)) {
        return function->kind() == arrow::compute::Function::SCALAR;        
    }
    return false;
}

bool arrow_compute_bound_is_satisfiable(BoundExpression bound) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    return expr->IsSatisfiable();
}

BoundExpression arrow_compute_get_bound_arg(BoundExpression bound, size_t idx) {
    auto expr = retrieve_instance<arrow::compute::Expression>(bound);
    if (auto* call = expr->call()) {
        return create_ref<arrow::compute::Expression>(std::make_shared<arrow::compute::Expression>(call->arguments[idx]));
    }
    return 0;
}

int arrow_compute_bound_expr_simplify_guarantee(BoundExpression expr, 
    const uint8_t* serialized_guarantee, const int serialized_len,
    BoundExpression* out) {

    auto guarantee_result = arrow::compute::Deserialize(std::make_shared<arrow::Buffer>(serialized_guarantee, serialized_len));
    if (!guarantee_result.ok()) {
        std::cerr << "msg: " << guarantee_result.status().message() << std::endl;
        return 0;
    }

    auto bound = retrieve_instance<arrow::compute::Expression>(expr);
    auto simplified = arrow::compute::SimplifyWithGuarantee(*bound, guarantee_result.MoveValueUnsafe());
    if (!simplified.ok()) {
        std::cerr << "msg: " << guarantee_result.status().message() << std::endl;
        return 0;
    }

    *out = create_ref<arrow::compute::Expression>(std::make_shared<arrow::compute::Expression>(simplified.MoveValueUnsafe()));
    return 0;
}

arrow::Status export_datum(const arrow::Datum& datum, struct ArrowComputeInputOutput* result) {
    switch (datum.kind()) {
    case arrow::Datum::RECORD_BATCH:
        return arrow::ExportRecordBatch(*datum.record_batch(), result->data, result->schema);
    case arrow::Datum::ARRAY:
        {
            auto status = arrow::ExportArray(*datum.make_array(), result->data);
            if (!status.ok()) {
                return status;
            }
            return arrow::ExportField(arrow::Field("array", datum.type()), result->schema);
        }
    case arrow::Datum::SCALAR:
        auto status = arrow::MakeArrayFromScalar(*datum.scalar(), 1);
        if (!status.ok()) {
            return status.status();
        }
        arrow::ExportField(arrow::Field("scalar", datum.type()), result->schema);
        return arrow::ExportArray(*status.MoveValueUnsafe(), result->data);
    }

    return arrow::Status::NotImplemented("only record batch, array and scalar is implemented");
}

int arrow_compute_execute_scalar_expr_schema(ExecContext ctx,
                                             struct ArrowSchema* full_schema, struct ArrowComputeInputOutput* partial_input,
                                             BoundExpression expr, struct ArrowComputeInputOutput* result) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    auto bound = retrieve_instance<arrow::compute::Expression>(expr);

    auto schema = arrow::ImportSchema(full_schema).ValueOrDie();
    auto input = arrow::ImportRecordBatch(partial_input->data, partial_input->schema);
    if (!input.ok()) {
        std::cerr << input.status().message() << std::endl;
        return 2;
    }        
    auto in = arrow::Datum(input.MoveValueUnsafe());

    auto output = arrow::compute::ExecuteScalarExpression(*bound, *schema, in, exec_context.get());
    if (!output.ok()) {
        std::cerr << output.status().message() << std::endl;
        return 2;
    }

    auto status = export_datum(output.ValueOrDie(), result);
    if (!status.ok()) {
        std::cerr << output.status().message() << std::endl;
        return 3;
    }    

    return 0;
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

int call_function(ExecContext ctx, const char* func_name, struct ArrowComputeInputOutput* args, 
                  struct ArrowComputeInputOutput* options, struct ArrowComputeInputOutput* results) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);

    auto input = arrow::ImportRecordBatch(args->data, args->schema);
    if (!input.ok()) {
        std::cerr << input.status().message() << std::endl;
        return 2;
    }        
    auto batch = input.MoveValueUnsafe();

    const auto& schema = batch->schema();
    std::vector<arrow::Datum> func_args;
    for (size_t i = 0; i < batch->num_columns(); ++i) {
        const auto& field = schema->field(i);
        const std::string datumtype = field->metadata()->Get("arrow::datum::type").ValueOr("");
        if (datumtype == "scalar") {
            func_args.emplace_back(batch->column(i)->GetScalar(0).ValueOrDie());
        } else if (datumtype == "array") {
            func_args.emplace_back(batch->column(i));
        } else if (datumtype == "record") {            
            auto status = arrow::RecordBatch::FromStructArray(batch->column(i));
            if (!status.ok()) {
                std::cerr << status.status().message() << std::endl;
                return 3;
            }
            func_args.emplace_back(status.MoveValueUnsafe());
        } else {
            return 4;
        }
    }

    auto options_scalar = arrow::ImportArray(options->data, options->schema).ValueOrDie()->GetScalar(0).ValueOrDie();    
    auto func_options = arrow::compute::internal::FunctionOptionsFromStructScalar(arrow::checked_cast<const arrow::StructScalar&>(*options_scalar)).ValueOrDie();

    auto resultstatus = arrow::compute::CallFunction(std::string(func_name), func_args, func_options.get(), exec_context.get());
    if (!resultstatus.ok()) {
        std::cerr << resultstatus.status().message() << std::endl;
        return 5;
    }

    auto status = export_datum(resultstatus.MoveValueUnsafe(), results);
    if (!status.ok()) {
        std::cerr << status.message() << std::endl;
        return 6;
    }

    return 0;
}
