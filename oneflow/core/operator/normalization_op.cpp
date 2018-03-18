#include "oneflow/core/operator/normalization_op.h"

namespace oneflow {

void NormalizationOp::InitFromOpConf() {
  CHECK_GT(op_conf().normalization_conf().epsilon(), 0.f);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollOtherBn("moving_mean");
  EnrollOtherBn("moving_variance");
  EnrollModelBn("beta");
  EnrollModelBn("gamma");
  EnrollDataTmpBn("normalized_in");
  EnrollDataTmpBn("inv_var");
  EnrollModelTmpBn("inv_elem_num");
  EnrollModelTmpBn("tmp_storage_for_sum");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("normalized_in") = *GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  BlobDesc blob_desc(Shape({1}), DataType::kFloat, false, false, 1);
  for (const auto& bn_in_op : {"moving_mean", "moving_variance", "beta",
                               "gamma", "inv_var", "inv_elem_num"}) {
    *GetBlobDesc4BnInOp(bn_in_op) = blob_desc;
  }
  int64_t tmp_storage_size =
      std::sqrt(GetBlobDesc4BnInOp("in")->shape().elem_cnt());
  *GetBlobDesc4BnInOp("tmp_storage_for_sum") = BlobDesc(
      Shape({tmp_storage_size + 1}), DataType::kFloat, false, false, 1);
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
