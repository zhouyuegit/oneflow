#include "oneflow/core/operator/sqrt_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SqrtGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sqrt_conf());
  EnrollInputBn("out");
  EnrollInputBn("out_diff");
  EnrollOutputBn("in_diff")->set_mutable_inplace_ibn("out_diff");
}

const PbMessage& SqrtGradOp::GetCustomizedConf() const { return op_conf().sqrt_grad_conf(); }

void SqrtGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("in_diff") = *GetBlobDesc4BnInOp("out_diff");
}

void SqrtGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kSqrtGradConf, SqrtGradOp);

}  // namespace oneflow
