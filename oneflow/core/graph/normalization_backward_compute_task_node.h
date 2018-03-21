#ifndef ONEFLOW_CORE_GRAPH_NORMALIZATION_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMALIZATION_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class NormalizationBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationBackwardCompTaskNode);
  NormalizationBackwardCompTaskNode() = default;
  ~NormalizationBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kNormalBackward; }

 private:
  void VirtualBuildExecGphAndBindOutDiffRegst() override;
  void VirtualBuildActivationDiffRegst() override;
  void VirtualBuildInDiffRegst() override;
  void VirtualConsumeRegstOnInEdge(TaskEdge* edge) override;
  void VirtualConsumeInRegst() override;
  void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) override;
  void VirtualProduceActivationDiff() override;
  void VirtualConsumeActivation(TaskEdge* edge) override;
  void VirtualInferBlobDescInActivationDiff() override;
  void VirtualBuildExtraRegsts() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMALIZATION_BACKWARD_COMPUTE_TASK_NODE_H_
