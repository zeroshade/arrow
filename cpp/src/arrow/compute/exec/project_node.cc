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

#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/util.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"

namespace arrow {

using internal::checked_cast;

namespace compute {
namespace {

class ProjectNode : public ExecNode {
 public:
  ProjectNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
              std::shared_ptr<Schema> output_schema, std::vector<Expression> exprs)
      : ExecNode(plan, std::move(inputs), /*input_labels=*/{"target"},
                 std::move(output_schema),
                 /*num_outputs=*/1),
        exprs_(std::move(exprs)) {}

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "ProjectNode"));

    const auto& project_options = checked_cast<const ProjectNodeOptions&>(options);
    auto exprs = project_options.expressions;
    auto names = project_options.names;

    if (names.size() == 0) {
      names.resize(exprs.size());
      for (size_t i = 0; i < exprs.size(); ++i) {
        names[i] = exprs[i].ToString();
      }
    }

    FieldVector fields(exprs.size());
    int i = 0;
    for (auto& expr : exprs) {
      if (!expr.IsBound()) {
        ARROW_ASSIGN_OR_RAISE(expr, expr.Bind(*inputs[0]->output_schema()));
      }
      fields[i] = field(std::move(names[i]), expr.type());
      ++i;
    }

    return plan->EmplaceNode<ProjectNode>(plan, std::move(inputs),
                                          schema(std::move(fields)), std::move(exprs));
  }

  const char* kind_name() override { return "ProjectNode"; }

  Result<ExecBatch> DoProject(const ExecBatch& target) {
    std::vector<Datum> values{exprs_.size()};
    for (size_t i = 0; i < exprs_.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(Expression simplified_expr,
                            SimplifyWithGuarantee(exprs_[i], target.guarantee));

      ARROW_ASSIGN_OR_RAISE(values[i], ExecuteScalarExpression(simplified_expr, target,
                                                               plan()->exec_context()));
    }
    return ExecBatch{std::move(values), target.length};
  }

  void InputReceived(ExecNode* input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    auto maybe_projected = DoProject(std::move(batch));
    if (ErrorIfNotOk(maybe_projected.status())) return;

    maybe_projected->guarantee = batch.guarantee;
    outputs_[0]->InputReceived(this, maybe_projected.MoveValueUnsafe());
  }

  void ErrorReceived(ExecNode* input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->ErrorReceived(this, std::move(error));
  }

  void InputFinished(ExecNode* input, int total_batches) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->InputFinished(this, total_batches);
  }

  Status StartProducing() override { return Status::OK(); }

  void PauseProducing(ExecNode* output) override {}

  void ResumeProducing(ExecNode* output) override {}

  void StopProducing(ExecNode* output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override { inputs_[0]->StopProducing(this); }

  Future<> finished() override { return inputs_[0]->finished(); }

 private:
  std::vector<Expression> exprs_;
};

}  // namespace

namespace internal {

void RegisterProjectNode(ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("project", ProjectNode::Make));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
