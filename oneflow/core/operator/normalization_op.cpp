#include "oneflow/core/operator/normalization_op.h"

namespace oneflow {

bool NormalizationOp::HasScaleOrCenter() const {
  const auto& normalization_conf = op_conf().normalization_conf();
  return normalization_conf.center() || normalization_conf.scale();
}

void NormalizationOp::InitFromOpConf() {
  const auto& normalization_conf = op_conf().normalization_conf();
  CHECK_GT(normalization_conf.epsilon(), 0.f);
  CHECK_GE(normalization_conf.momentum(), 0);
  CHECK_LE(normalization_conf.momentum(), 1);
  EnrollInputBn("inputs");
  EnrollOutputBn("outputs");
  EnrollDataTmpBn("mean");
  EnrollDataTmpBn("variance");
  EnrollOtherBn("moving_mean");
  EnrollOtherBn("moving_variance");
  if (normalization_conf.center()) { EnrollModelBn("beta"); }
  if (normalization_conf.scale()) { EnrollModelBn("gamma"); }
  if (HasScaleOrCenter()) { EnrollDataTmpBn("normalized_inputs"); }
  EnrollDataTmpBn("inv_var");
  EnrollModelTmpBn("momentum");
  EnrollModelTmpBn("inv_elem_num");
  EnrollModelTmpBn("tmp_storage_for_sum");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& normalization_conf = op_conf().normalization_conf();
  if (HasScaleOrCenter()) {
    *GetBlobDesc4BnInOp("normalized_inputs") = *GetBlobDesc4BnInOp("inputs");
  }
  *GetBlobDesc4BnInOp("outputs") = *GetBlobDesc4BnInOp("inputs");
  BlobDesc blob_desc(Shape({1}), DataType::kFloat, false, false, 1);
  std::list<std::string> scalar_blob_names = {
      "mean",    "variance", "moving_mean", "moving_variance",
      "inv_var", "momentum", "inv_elem_num"};
  if (normalization_conf.center()) { scalar_blob_names.push_back("beta"); }
  if (normalization_conf.scale()) { scalar_blob_names.push_back("gamma"); }
  for (const auto& bn_in_op : scalar_blob_names) {
    *GetBlobDesc4BnInOp(bn_in_op) = blob_desc;
  }
  int64_t tmp_storage_size =
      std::sqrt(GetBlobDesc4BnInOp("inputs")->shape().elem_cnt());
  *GetBlobDesc4BnInOp("tmp_storage_for_sum") = BlobDesc(
      Shape({tmp_storage_size + 1}), DataType::kFloat, false, false, 1);
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
