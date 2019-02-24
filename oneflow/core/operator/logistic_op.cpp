#include "oneflow/core/operator/logistic_op.h"

namespace oneflow {

void LogisticOp::InitFromOpConf() {
  CHECK(op_conf().has_logistic_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& LogisticOp::GetCustomizedConf() const { return op_conf().logistic_conf(); }

void LogisticOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kLogisticConf, LogisticOp);

}  // namespace oneflow
