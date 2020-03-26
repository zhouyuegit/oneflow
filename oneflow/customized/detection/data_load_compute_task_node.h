#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class DetectionDataLoadCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DetectionDataLoadCompTaskNode);
  DetectionDataLoadCompTaskNode() = default;
  ~DetectionDataLoadCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  // bool IsMeaningLess() override { return false; }

  TaskType GetTaskType() const override { return TaskType::kDetectionDataLoad; }
  bool IsIndependent() const override { return true; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_COMPUTE_TASK_NODE_H_
