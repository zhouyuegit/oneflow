#include "oneflow/core/operator/log_op.h"

namespace oneflow {

void LogOp::InitFromOpConf() {
  CHECK(op_conf().has_log_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& LogOp::GetCustomizedConf() const { return op_conf().log_conf(); }

void LogOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kLogConf, LogOp);

}  // namespace oneflow
