#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

extern int kThisMachineId;

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

MachineCtx::MachineCtx() : this_machine_id_(kThisMachineId) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
