#include "oneflow/core/operator/where_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void WhereOp::InitFromOpConf() {
  CHECK(op_conf().has_where_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("x");
  EnrollInputBn("y");
  EnrollOutputBn("out");
}

const PbMessage& WhereOp::GetCustomizedConf() const { return op_conf().where_conf(); }

void WhereOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("x");
}

void WhereOp::GetSbpSignature(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("condition", 0)
      .Split("x", 0)
      .Split("y", 0)
      .Split("out", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
