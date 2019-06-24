#include "oneflow/core/operator/where_x_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void WhereXGradOp::InitFromOpConf() {
  CHECK(op_conf().has_where_grad_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("out_diff");
  EnrollOutputBn("x_diff");
}

const PbMessage& WhereXGradOp::GetCustomizedConf() const { return op_conf().where_grad_conf(); }

void WhereXGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("x_diff") = *GetBlobDesc4BnInOp("out_diff");
}

void WhereXGradOp::GetSbpSignature(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("condition", 0)
      .Split("out_diff", 0)
      .Split("x_diff", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kWhereXGradConf, WhereXGradOp);

}  // namespace oneflow
