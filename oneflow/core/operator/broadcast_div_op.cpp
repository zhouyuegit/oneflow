#include "oneflow/core/operator/broadcast_div_op.h"

namespace oneflow {

const PbMessage& BroadcastDivOp::GetCustomizedConf() const {
  return op_conf().broadcast_div_conf();
}

Maybe<void> BroadcastDivOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("a").Broadcast("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastDivConf, BroadcastDivOp);

}  // namespace oneflow
