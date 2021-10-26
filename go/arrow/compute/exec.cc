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

// +build ccalloc ccexec

#include "exec.h"

#include "../memory/internal/cgoalloc/helpers.h"
#include "arrow/array.h"
#include "arrow/array/util.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/function.h"
#include "arrow/compute/registry.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include <iostream>

const int64_t kDefaultExecChunk = arrow::compute::kDefaultExecChunksize;
const int arrow_shape_any = static_cast<int>(arrow::ValueDescr::ANY);
const int arrow_shape_array = static_cast<int>(arrow::ValueDescr::ARRAY);
const int arrow_shape_scalar = static_cast<int>(arrow::ValueDescr::SCALAR);

bool arrow_compute_function_is_scalar(const char* funcname) {
  if (auto function =
          arrow::compute::GetFunctionRegistry()->GetFunction(funcname).ValueOr(nullptr)) {
    return function->kind() == arrow::compute::Function::SCALAR;
  }
  return false;
}

ExecContext arrow_compute_default_context() {
  std::shared_ptr<arrow::compute::ExecContext> ctx(arrow::compute::default_exec_context(),
                                                   [](arrow::compute::ExecContext*) {});
  return create_ref<arrow::compute::ExecContext>(ctx);
}

void arrow_compute_release_context(ExecContext ctx) {
  release_ref<arrow::compute::ExecContext>(ctx);
}

int64_t arrow_compute_get_exec_chunksize(ExecContext ctx) {
  auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
  return exec_context->exec_chunksize();
}

void arrow_compute_set_exec_chunksize(ExecContext ctx, int64_t chunksize) {
  auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
  return exec_context->set_exec_chunksize(chunksize);
}

struct bound_expr_holder {
  arrow::Status last_status;
  arrow::compute::Expression expr;
  std::shared_ptr<arrow::Buffer> serialized;
};

arrow_postbind_info arrow_compute_bind_expr(ExecContext ctx, struct ArrowSchema* schema,
                                            const uint8_t* serialized_expr,
                                            const int serialized_len) {
  arrow_postbind_info ret{0, nullptr, 0, nullptr, -1, nullptr, -1};
  auto output = std::make_shared<bound_expr_holder>();  
  ret.bound = create_ref(output);

  arrow::compute::ExecContext* ectx = nullptr;
  if (ctx != 0) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    ectx = exec_context.get();
  }

  auto expr = arrow::compute::Deserialize(
      std::make_shared<arrow::Buffer>(serialized_expr, serialized_len));
  if (!expr.ok()) {
    output->last_status = expr.status();
    ret.status = output->last_status.message().c_str();
    return ret;
  }

  auto schema_res = arrow::ImportSchema(schema);
  if (!schema_res.ok()) {
    output->last_status = schema_res.status();
    ret.status = output->last_status.message().c_str();
    return ret;
  }

  auto expr_result = expr.ValueUnsafe().Bind(*schema_res.ValueUnsafe(), ectx);
  if (!expr_result.ok()) {
    output->last_status = expr_result.status();
    ret.status = output->last_status.message().c_str();
    return ret;
  }

  output->expr = std::move(expr_result.MoveValueUnsafe());
  ret.shape = static_cast<int>(output->expr.descr().shape);
  if (output->expr.literal() == nullptr) {
    ret.type = (struct ArrowSchema*)(malloc(sizeof(struct ArrowSchema)));    
    output->last_status = arrow::ExportType(*(output->expr.type()), ret.type);
    if (!output->last_status.ok()) {
      ret.status = output->last_status.message().c_str();
    }
  }

  if (auto* param = output->expr.parameter()) {
    ret.index = param->index;
  }

  if (auto* call = output->expr.call()) {
    auto serialize_result = arrow::compute::Serialize(output->expr);
    if (!serialize_result.ok()) {
      output->last_status = serialize_result.status();
      ret.status = output->last_status.message().c_str();
      return ret;
    }

    output->serialized = std::move(serialize_result.MoveValueUnsafe());
    ret.serialized_data = output->serialized->data();
    ret.serialized_len = output->serialized->size();
  }

  return ret;
}

const char* arrow_compute_expr_last_status(BoundExpression bound) {
  auto holder = retrieve_instance<bound_expr_holder>(bound);
  return holder->last_status.message().c_str();
}

bool arrow_compute_bound_expr_type(BoundExpression bound, struct ArrowSchema* out) {
  auto holder = retrieve_instance<bound_expr_holder>(bound);
  holder->last_status = arrow::ExportType(*(holder->expr.type()), out);
  return holder->last_status.ok();
}

arrow_postbind_info arrow_compute_get_bound_arg(BoundExpression bound, size_t idx) {
  arrow_postbind_info ret{0, nullptr, -1, nullptr, -1};
  auto holder = retrieve_instance<bound_expr_holder>(bound);
  if (auto* call = holder->expr.call()) {
    auto out = std::make_shared<bound_expr_holder>();
    out->expr = call->arguments[idx];
    ret.bound = create_ref(out);
    ret.shape = static_cast<int>(out->expr.descr().shape);
    ret.type = (struct ArrowSchema*)(malloc(sizeof(struct ArrowSchema)));
    out->last_status = arrow::ExportType(*(out->expr.type()), ret.type);
    if (!out->last_status.ok()) {
      ret.status = out->last_status.message().c_str();
    }

    if (auto* param = out->expr.parameter()) {
      ret.index = param->index;
    }
  }
  return ret;
}

void arrow_compute_bound_expr_release(BoundExpression bound) {
  release_ref<bound_expr_holder>(bound);
}

arrow::Result<arrow::Datum> import_datum(struct ArrowDatum* imported) {
  auto kind = static_cast<arrow::Datum::Kind>(imported->datum_type);
  if (kind == arrow::Datum::NONE) {
    return arrow::Datum{};
  }

  switch (kind) {
    case arrow::Datum::SCALAR: {
      ARROW_ASSIGN_OR_RAISE(auto arr,
                            arrow::ImportArray(imported->data[0], imported->schema));
      ARROW_ASSIGN_OR_RAISE(auto scalar, arr->GetScalar(0));
      return arrow::Datum(scalar);
    }
    case arrow::Datum::ARRAY: {
      ARROW_ASSIGN_OR_RAISE(auto arr,
                            arrow::ImportArray(imported->data[0], imported->schema));
      return arrow::Datum{arr};
    }
    case arrow::Datum::CHUNKED_ARRAY: {
      arrow::ArrayVector chunks;
      ARROW_ASSIGN_OR_RAISE(auto type, arrow::ImportType(imported->schema));
      for (int i = 0; i < imported->num_data; ++i) {
        ARROW_ASSIGN_OR_RAISE(auto arr, arrow::ImportArray(imported->data[i], type));
        chunks.emplace_back(std::move(arr));
      }
      return arrow::Datum{arrow::ChunkedArray{chunks}};
    }
    case arrow::Datum::RECORD_BATCH: {
      ARROW_ASSIGN_OR_RAISE(
          auto arr, arrow::ImportRecordBatch(imported->data[0], imported->schema));
      return arrow::Datum{std::move(arr)};
    }
    case arrow::Datum::TABLE: {
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      ARROW_ASSIGN_OR_RAISE(auto schema, arrow::ImportSchema(imported->schema));
      for (int i = 0; i < imported->num_data; ++i) {
        ARROW_ASSIGN_OR_RAISE(auto batch,
                              arrow::ImportRecordBatch(imported->data[i], schema));
        batches.emplace_back(std::move(batch));
      }
      ARROW_ASSIGN_OR_RAISE(auto tbl, arrow::Table::FromRecordBatches(schema, batches));
      return arrow::Datum{std::move(tbl)};
    }
    case arrow::Datum::COLLECTION:
      throw std::runtime_error("not implemented");
  }

  return arrow::Datum{};
}

arrow::Result<struct ArrowDatum*> export_datum(const arrow::Datum& datum) {
  arrow::ArrayVector arraylist;

  switch (datum.kind()) {
    case arrow::Datum::NONE:
      break;
    case arrow::Datum::SCALAR: {
      ARROW_ASSIGN_OR_RAISE(auto scalar, arrow::MakeArrayFromScalar(*datum.scalar(), 1));
      arraylist.push_back(std::move(scalar));
      break;
    }
    case arrow::Datum::ARRAY:
    case arrow::Datum::CHUNKED_ARRAY:
      arraylist = std::move(datum.chunks());
      break;
    case arrow::Datum::RECORD_BATCH: {
      ARROW_ASSIGN_OR_RAISE(auto struct_arr, datum.record_batch()->ToStructArray());
      arraylist.push_back(std::dynamic_pointer_cast<arrow::Array>(struct_arr));
      break;
    }
    case arrow::Datum::TABLE: {
      arrow::TableBatchReader rdr(*datum.table());
      std::shared_ptr<arrow::RecordBatch> batch;
      while (true) {
        RETURN_NOT_OK(rdr.ReadNext(&batch));
        if (!batch) {
          break;
        }

        ARROW_ASSIGN_OR_RAISE(auto struct_arr, datum.record_batch()->ToStructArray());
        arraylist.push_back(std::dynamic_pointer_cast<arrow::Array>(struct_arr));
      }
      break;
    }
    case arrow::Datum::COLLECTION:
      throw std::runtime_error("not implemented");
  }

  struct ArrowDatum* output = (struct ArrowDatum*)(malloc(sizeof(struct ArrowDatum)));
  output->datum_type = static_cast<int>(datum.kind());
  output->num_data = arraylist.size();

  if (output->num_data == 0) {
    return output;
  }

  output->schema = (struct ArrowSchema*)(malloc(sizeof(struct ArrowSchema)));
  RETURN_NOT_OK(arrow::ExportType(*arraylist[0]->type(), output->schema));

  auto arrowoutput =
      (struct ArrowArray*)(malloc(sizeof(struct ArrowArray) * arraylist.size()));
  output->data =
      (struct ArrowArray**)(malloc(sizeof(struct ArrowArray*) * arraylist.size()));

  for (size_t i = 0; i < arraylist.size(); ++i) {
    output->data[i] = &arrowoutput[i];
    RETURN_NOT_OK(arrow::ExportArray(*arraylist[i], output->data[i]));
  }

  return output;
}

struct ArrowDatum* arrow_compute_exec_scalar_expr(BoundExpression bound,
                                                  struct ArrowSchema* full_schema,
                                                  struct ArrowDatum* input,
                                                  ExecContext ctx) {
  auto holder = retrieve_instance<bound_expr_holder>(bound);
  arrow::compute::ExecContext* ectx = nullptr;
  if (ctx != 0) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    ectx = exec_context.get();
  }

  auto schema_status = arrow::ImportSchema(full_schema);
  if (!schema_status.ok()) {
    throw std::runtime_error(schema_status.status().message());
  }

  auto datum = import_datum(input).ValueOrDie();
  auto output = arrow::compute::ExecuteScalarExpression(
      holder->expr, *(schema_status.ValueOrDie()), datum, ectx);
  return export_datum(output.ValueOrDie()).ValueOrDie();
}

void freeArrowDatum(struct ArrowDatum* in) {
  if (in == nullptr) {
    return;
  }

  if (in->schema != nullptr) {
    ArrowSchemaRelease(in->schema);
    free(in->schema);
  }

  if (in->num_data > 0) {
    for (int i = 0; i < in->num_data; ++i) {
      ArrowArrayRelease(in->data[i]);
    }
    free(in->data[0]);
    free(in->data);
  }

  free(in);
}

struct ArrowDatum* arrow_compute_call_function(
    ExecContext ctx, const char* func_name, struct ArrowDatum** args, const int num_args,
    const char* options_type, const uint8_t* serialized_opts, const int serialized_len) {
  arrow::compute::ExecContext* ectx = nullptr;
  if (ctx != 0) {
    auto exec_context = retrieve_instance<arrow::compute::ExecContext>(ctx);
    ectx = exec_context.get();
  }

  std::vector<arrow::Datum> datum_args;
  for (int i = 0; i < num_args; ++i) {
    datum_args.emplace_back(std::move(import_datum(args[i]).ValueOrDie()));
  }

  std::unique_ptr<arrow::compute::FunctionOptions> options;
  if (serialized_opts) {
    options = arrow::compute::FunctionOptions::Deserialize(
                  options_type, arrow::Buffer{serialized_opts, serialized_len})
                  .ValueOrDie();
  }
  auto output = arrow::compute::CallFunction(func_name, datum_args, options.get(), ectx);
  return export_datum(output.ValueOrDie()).ValueOrDie();
}
