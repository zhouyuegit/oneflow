#include "oneflow/core/operator/exp_op.h"

namespace oneflow {

void ExpOp::InitFromOpConf() {
  CHECK(op_conf().has_exp_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ExpOp::GetCustomizedConf() const { return op_conf().exp_conf(); }

void ExpOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kExpConf, ExpOp);

}  // namespace oneflow
