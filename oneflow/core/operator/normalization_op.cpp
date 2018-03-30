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
  // TODO: axis list
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("new_mean");
  EnrollDataTmpBn("new_variance");
  EnrollForwardModelBn("moving_mean");
  EnrollForwardModelBn("moving_variance");

  if (normalization_conf.center()) { EnrollModelBn("beta"); }
  if (normalization_conf.scale()) { EnrollModelBn("gamma"); }
  if (HasScaleOrCenter()) { EnrollDataTmpBn("normalized_in"); }
  EnrollDataTmpBn("inv_var");
  EnrollModelTmpBn("tmp_storage_for_sum");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& normalization_conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(),
           Global<JobDesc>::Get()->DefaultDataType());
  int32_t axis = normalization_conf.axis();
  int64_t normalize_elem_cnt = in_blob_desc->shape().elem_cnt() / in_blob_desc->shape().At(axis);
  CHECK(axis < in_blob_desc->shape().NumAxes());
  if (HasScaleOrCenter()) {
    *GetBlobDesc4BnInOp("normalized_in") = *in_blob_desc;
  }
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc blob_desc(Shape({in_blob_desc->shape().At(axis)}), in_blob_desc->data_type(), false, false,
                     1);
  std::list<std::string> blob_names = {"moving_mean", "moving_variance",
                                              "inv_var"};
  std::list<std::string> bns_needless_in_predict = {"new_mean", "new_variance"};
  if (normalization_conf.center()) { blob_names.push_back("beta"); }
  if (normalization_conf.scale()) { blob_names.push_back("gamma"); }
  if (Global<JobDesc>::Get()->IsTrain()) {
    for (const std::string& bn : bns_needless_in_predict) {
      blob_names.push_back(bn);
    }
  }
  for (const auto& bn_in_op : blob_names) {
    *GetBlobDesc4BnInOp(bn_in_op)->mut_shape() = *blob_desc;
  }
  int64_t tmp_storage_size =
      std::sqrt(normalize_elem_cnt);
  GetBlobDesc4BnInOp("tmp_storage_for_sum")->set_data_type(in_blob_desc->data_type());
  GetBlobDesc4BnInOp("tmp_storage_for_sum")->mut_shape() =
      Shape({tmp_storage_size + 1});
}

void NormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext*, KernelConf* kernel_conf) const {
  const auto& normalization_conf = op_conf().normalization_conf();
  int32_t axis = normalization_conf.axis();
  int64_t normalize_elem_cnt = in_blob_desc->shape().elem_cnt() / in_blob_desc->shape().At(axis);
  kernel_conf->mutable_normalization_conf()->set_inv_norm_elem_cnt(
      1.0 / normalize_elem_cnt);
}

void NormalizationOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
