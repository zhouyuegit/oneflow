#include "oneflow/core/job_completer/job_completer_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

void FixVariableUtil::try_fix_variable_with_default_initialize_with_snapshot_path(
    const Job& job, OperatorConf* variable_op_conf, std::string model_name) {
  const bool has_default_initialize_with_snapshot_path =
      job.job_conf().has_default_initialize_with_snapshot_path();
  if (has_default_initialize_with_snapshot_path == false) { return; }
  VariableOpConf* variable_conf = variable_op_conf->mutable_variable_conf();
  const std::string& op_name = variable_op_conf->name();
  const std::string file_path =
      JoinPath(job.job_conf().default_initialize_with_snapshot_path(), op_name, model_name);
  if (SnapshotFS()->FileExists(file_path)) {
    const std::string dir_path =
        JoinPath(job.job_conf().default_initialize_with_snapshot_path(), op_name);
    variable_conf->mutable_initialize_with_snapshot()->set_path(JoinPath(dir_path));
    variable_conf->mutable_initialize_with_snapshot()->set_key(model_name);
  } else {
    LOG(ERROR) << file_path << " not found, will be initialized";
  }
}

}  // namespace oneflow
