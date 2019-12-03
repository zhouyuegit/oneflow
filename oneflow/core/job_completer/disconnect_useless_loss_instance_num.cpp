#include "oneflow/core/job_completer/disconnect_useless_loss_instance_num.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool IsOpConstantOne(const OperatorConf& op_conf) {
  if (!op_conf.has_constant_conf()) { return false; }
  const InitializerConf& initializer_conf = op_conf.constant_conf().initializer();
  if (initializer_conf.has_constant_int_conf()) {
    return initializer_conf.constant_int_conf().value() == 1;
  } else if (initializer_conf.has_constant_conf()) {
    return initializer_conf.constant_conf().value() == 1;
  } else {
    UNIMPLEMENTED();
    return false;
  }
}

void DisconnectUselessLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (op_conf.has_lazy_adam_model_update_conf()) {
      const std::string& total_instance_num_diff_lbn =
          op_conf.lazy_adam_model_update_conf().total_instance_num_diff();
      if (total_instance_num_diff_lbn.empty()) { return; }
      const OperatorConf& producer_op_conf =
          op_graph.OpNode4OpName(GenLogicalBlobId(total_instance_num_diff_lbn).op_name())
              ->op()
              .op_conf();
      if (!IsOpConstantOne(producer_op_conf)) { return; }
      OperatorConf new_op_conf = op_node->op().op_conf();
      new_op_conf.mutable_lazy_adam_model_update_conf()->clear_total_instance_num_diff();
      job_builder->MutOpsOnlyOnce({new_op_conf});
      return;
    } else {
      return;
    }
  });
}

}  // namespace oneflow
