#include "oneflow/core/operator/normalization_op.h"

namespace oneflow {

void NormalizationOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("moving_mean");
  EnrollModelBn("moving_variance");
  EnrollModelBn("beta");
  EnrollModelBn("gamma");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  BlobDesc blob_desc(Shape({1}), DataType::kFloat, false, false, 1);
  for (const auto& bn_in_op :
       {"moving_mean", "moving_variance", "beta", "gamma"}) {
    *GetBlobDesc4BnInOp(bn_in_op) = blob_desc;
  }
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
