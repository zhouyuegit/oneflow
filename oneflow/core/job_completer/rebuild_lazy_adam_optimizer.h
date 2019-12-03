#ifndef ONEFLOW_CORE_JOB_COMPLETER_REBUILD_LAZY_ADAM_OPTIMIZER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_REBUILD_LAZY_ADAM_OPTIMIZER_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;
class Job;

void RebuildLazyAdamOptimizer(const OpGraph& op_graph, JobBuilder* job_builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_REBUILD_LAZY_ADAM_OPTIMIZER_H_
