#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/normalization_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NormalizationForwardCompTaskNode::VirtualConsumeRegstOnInEdge(TaskEdge* edge) {
  if (edge->src_node()->GetTaskType() == TaskType::kNormalizationMdUpdt) {
      ConsumeRegst("norm_model", edge->GetSoleRegst());
  } else {
    ConsumeRegst("in", edge->GetSoleRegst());
  }
}

void NormalizationForwardCompTaskNode::VirtualProduceRegstOnOutEdge(TaskEdge* edge) {
  if (edge->dst_node()->GetTaskType() == TaskType::kNormalizationMdUpdt) {
    edge->AddRegst("norm_acc", ProduceRegst("norm_acc"));
  } else {
    edge->AddRegst("out", GetProducedRegst("out"));
    if (IsBackwardTaskType(edge->dst_node()->GetTaskType())) {
      edge->AddRegst("activation", ProduceRegst("activation"));
      edge->AddRegst("data_tmp", ProduceRegst("data_tmp"));
    }
  }
}

void NormalizationForwardCompTaskNode::VirtualBuildExecGphStructAndBindInRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  ExecNode* cur_node = mut_exec_gph().SoleNode();
    for (const std::string& ibn : cur_node->op()->input_bns()) {
        cur_node->BindBnInOpAndRegst(ibn, in_regst);
      }
}

void NormalizationForwardCompTaskNode::VirtualBuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* cur_node = mut_exec_gph().SoleNode();
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      out_regst->AddLbn(lbn);
      cur_node->BindBnInOpAndRegst(obn, out_regst);
    }
}

bool NormalizationForwardCompTaskNode::IsReadyForBuild() {
  return GetConsumedRegst("in")->IsLocked();
}

}  // namespace oneflow
