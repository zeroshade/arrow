
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

#include "arrow/compute/exec/exec_plan.h"

#include <mutex>

#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"
#include "arrow/util/optional.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/unreachable.h"

namespace arrow {

using internal::checked_cast;

namespace compute {
namespace {

class SinkNode : public ExecNode {
 public:
  SinkNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
           AsyncGenerator<util::optional<ExecBatch>>* generator)
      : ExecNode(plan, std::move(inputs), {"collected"}, {},
                 /*num_outputs=*/0),
        producer_(MakeProducer(generator)) {}

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "SinkNode"));

    const auto& sink_options = checked_cast<const SinkNodeOptions&>(options);
    return plan->EmplaceNode<SinkNode>(plan, std::move(inputs), sink_options.generator);
  }

  static PushGenerator<util::optional<ExecBatch>>::Producer MakeProducer(
      AsyncGenerator<util::optional<ExecBatch>>* out_gen) {
    PushGenerator<util::optional<ExecBatch>> push_gen;
    auto out = push_gen.producer();
    *out_gen = [push_gen] {
      // Awful workaround for MSVC 19.0 (Visual Studio 2015) bug.
      // For some types including Future<optional<ExecBatch>>,
      // std::is_convertible<T, T>::value will be false causing
      // SFINAE exclusion of the std::function constructor we need.
      // Definining a convertible (but distinct) type soothes the
      // faulty trait.
      struct ConvertibleToFuture {
        operator Future<util::optional<ExecBatch>>() && {  // NOLINT runtime/explicit
          return std::move(ret);
        }
        Future<util::optional<ExecBatch>> ret;
      };

      return ConvertibleToFuture{push_gen()};
    };
    return out;
  }

  const char* kind_name() override { return "SinkNode"; }

  Status StartProducing() override {
    finished_ = Future<>::Make();
    return Status::OK();
  }

  // sink nodes have no outputs from which to feel backpressure
  [[noreturn]] static void NoOutputs() {
    Unreachable("no outputs; this should never be called");
  }
  [[noreturn]] void ResumeProducing(ExecNode* output) override { NoOutputs(); }
  [[noreturn]] void PauseProducing(ExecNode* output) override { NoOutputs(); }
  [[noreturn]] void StopProducing(ExecNode* output) override { NoOutputs(); }

  void StopProducing() override {
    Finish();
    inputs_[0]->StopProducing(this);
  }

  Future<> finished() override { return finished_; }

  void InputReceived(ExecNode* input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    bool did_push = producer_.Push(std::move(batch));
    if (!did_push) return;  // producer_ was Closed already

    if (input_counter_.Increment()) {
      Finish();
    }
  }

  void ErrorReceived(ExecNode* input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);

    producer_.Push(std::move(error));

    if (input_counter_.Cancel()) {
      Finish();
    }
    inputs_[0]->StopProducing(this);
  }

  void InputFinished(ExecNode* input, int total_batches) override {
    if (input_counter_.SetTotal(total_batches)) {
      Finish();
    }
  }

 protected:
  virtual void Finish() {
    if (producer_.Close()) {
      finished_.MarkFinished();
    }
  }

  AtomicCounter input_counter_;
  Future<> finished_ = Future<>::MakeFinished();

  PushGenerator<util::optional<ExecBatch>>::Producer producer_;
};

// A sink node that accumulates inputs, then sorts them before emitting them.
struct OrderBySinkNode final : public SinkNode {
  OrderBySinkNode(ExecPlan* plan, std::vector<ExecNode*> inputs, SortOptions sort_options,
                  AsyncGenerator<util::optional<ExecBatch>>* generator)
      : SinkNode(plan, std::move(inputs), generator),
        sort_options_(std::move(sort_options)) {}

  const char* kind_name() override { return "OrderBySinkNode"; }

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "OrderBySinkNode"));

    const auto& sink_options = checked_cast<const OrderBySinkNodeOptions&>(options);
    return plan->EmplaceNode<OrderBySinkNode>(
        plan, std::move(inputs), sink_options.sort_options, sink_options.generator);
  }

  void InputReceived(ExecNode* input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    // Accumulate data
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto maybe_batch = batch.ToRecordBatch(inputs_[0]->output_schema(),
                                             plan()->exec_context()->memory_pool());
      if (ErrorIfNotOk(maybe_batch.status())) return;
      batches_.push_back(maybe_batch.MoveValueUnsafe());
    }

    if (input_counter_.Increment()) {
      Finish();
    }
  }

 protected:
  Status DoFinish() {
    Datum sorted;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ARROW_ASSIGN_OR_RAISE(
          auto table,
          Table::FromRecordBatches(inputs_[0]->output_schema(), std::move(batches_)));
      ARROW_ASSIGN_OR_RAISE(auto indices,
                            SortIndices(table, sort_options_, plan()->exec_context()));
      ARROW_ASSIGN_OR_RAISE(sorted, Take(table, indices, TakeOptions::NoBoundsCheck(),
                                         plan()->exec_context()));
    }
    TableBatchReader reader(*sorted.table());
    while (true) {
      std::shared_ptr<RecordBatch> batch;
      RETURN_NOT_OK(reader.ReadNext(&batch));
      if (!batch) break;
      bool did_push = producer_.Push(ExecBatch(*batch));
      if (!did_push) break;  // producer_ was Closed already
    }
    return Status::OK();
  }

  void Finish() override {
    Status st = DoFinish();
    if (ErrorIfNotOk(st)) {
      producer_.Push(std::move(st));
    }
    SinkNode::Finish();
  }

 private:
  SortOptions sort_options_;
  std::mutex mutex_;
  std::vector<std::shared_ptr<RecordBatch>> batches_;
};

}  // namespace

namespace internal {

void RegisterSinkNode(ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("order_by_sink", OrderBySinkNode::Make));
  DCHECK_OK(registry->AddFactory("sink", SinkNode::Make));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
