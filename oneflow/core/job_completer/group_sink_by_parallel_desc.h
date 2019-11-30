#ifndef ONEFLOW_CORE_JOB_COMPLETER_GROUP_SINK_BY_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_COMPLETER_GROUP_SINK_BY_PARALLEL_DESC_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;
class Job;

void GroupSinkByParallelDesc(const OpGraph& op_graph, JobBuilder* job_builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_GROUP_SINK_BY_PARALLEL_DESC_H_
