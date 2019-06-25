#include "oneflow/core/operator/l2_normalize_grad_op.h"

namespace oneflow {

void L2NormalizeGradOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_grad_conf());
  EnrollInputBn("out");
  EnrollInputBn("square_x_sum");
  EnrollInputBn("out_diff");
  EnrollOutputBn("in_diff")->set_mutable_inplace_ibn("out_diff");
}

const PbMessage& L2NormalizeGradOp::GetCustomizedConf() const {
  return op_conf().l2_normalize_grad_conf();
}

void L2NormalizeGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const L2NormalizeGradOpConf& conf = op_conf().l2_normalize_grad_conf();
  const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp("out_diff");
  int32_t axis_num = out_diff_blob_desc->shape().NumAxes();
  int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + axis_num;
  CHECK_GE(axis, 0);
  CHECK_LT(axis, axis_num);
  CHECK_GT(conf.epsilon(), 0);
  *GetBlobDesc4BnInOp("in_diff") = *out_diff_blob_desc;
}

void L2NormalizeGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("out_diff", 0)
      .Split("out", 0)
      .Split("square_x_sum", 0)
      .Split("in_diff", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kL2NormalizeGradConf, L2NormalizeGradOp);

}  // namespace oneflow
