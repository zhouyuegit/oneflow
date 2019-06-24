#include "oneflow/core/operator/where_y_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void WhereYGradOp::InitFromOpConf() {
  CHECK(op_conf().has_where_grad_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("out_diff");
  EnrollOutputBn("y_diff");
}

const PbMessage& WhereYGradOp::GetCustomizedConf() const { return op_conf().where_y_grad_conf(); }

void WhereYGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("y_diff") = *GetBlobDesc4BnInOp("out_diff");
}

void WhereYGradOp::GetSbpSignature(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("condition", 0)
      .Split("out_diff", 0)
      .Split("y_diff", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kWhereYGradConf, WhereYGradOp);

}  // namespace oneflow
