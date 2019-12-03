#ifndef ONEFLOW_CORE_JOB_COMPLETER_DISCONNECT_USELESS_LOSS_INSTANCE_NUM_H_
#define ONEFLOW_CORE_JOB_COMPLETER_DISCONNECT_USELESS_LOSS_INSTANCE_NUM_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;
class Job;

void DisconnectUselessLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_DISCONNECT_USELESS_LOSS_INSTANCE_NUM_H_
