#include "oneflow/core/operator/leaky_relu_op.h"

namespace oneflow {

void LeakyReluOp::InitFromOpConf() {
  CHECK(op_conf().has_leaky_relu_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& LeakyReluOp::GetCustomizedConf() const { return op_conf().leaky_relu_conf(); }

void LeakyReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kLeakyReluConf, LeakyReluOp);

}  // namespace oneflow
