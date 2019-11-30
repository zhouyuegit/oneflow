#include "oneflow/core/job_completer/group_sink_by_parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void GroupSinkByParallelDesc(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<const ParallelDesc&, std::vector<std::string>> parallel_desc2sink_op_name;
  op_graph.ForEachNode([&](OpNode* op_node) {
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt > 0) { return; }
    parallel_desc2sink_op_name[op_node->parallel_desc()].push_back(op_node->op().op_name());
  });
  for (const auto& pair : parallel_desc2sink_op_name) {
    const ParallelDesc& pd = pair.first;
    const std::vector<std::string>& op_names = pair.second;
    if (op_names.size() <= 1) { continue; }
    OperatorConf nop_op_conf{};
    nop_op_conf.set_name("System-Nop-" + NewUniqueId());
    for (const std::string& op_name : op_names) {
      *nop_op_conf.mutable_ctrl_in_op_name()->Add() = op_name;
    }
    job_builder->AddOps(pd.parallel_conf(), {nop_op_conf});
  }
}

}  // namespace oneflow
