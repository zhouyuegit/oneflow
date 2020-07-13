#ifndef ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_UTIL_H
#define ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_UTIL_H
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

struct FixVariableUtil {
 public:
  FixVariableUtil() = delete;
  ~FixVariableUtil() = delete;
  static void try_fix_variable_with_default_initialize_with_snapshot_path(
      const Job& job, OperatorConf* variable_op_conf, std::string blob_name);
};

}  // namespace oneflow

#endif ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_UTIL_H
