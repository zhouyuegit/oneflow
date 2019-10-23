#ifndef ONEFLOW_CORE_JOB_COMPLETER_INDEXED_SLICES_OPTIMIZER_REWRITE_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_INDEXED_SLICES_OPTIMIZER_REWRITE_PASS_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;

class IndexedSlicesOptimizerRewritePass final {
 public:
  IndexedSlicesOptimizerRewritePass() = default;
  ~IndexedSlicesOptimizerRewritePass() = default;
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_INDEXED_SLICES_OPTIMIZER_REWRITE_PASS_H_
