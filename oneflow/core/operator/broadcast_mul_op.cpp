#include "oneflow/core/operator/broadcast_mul_op.h"

namespace oneflow {

const PbMessage& BroadcastMulOp::GetCustomizedConf() const {
  return op_conf().broadcast_mul_conf();
}

Maybe<void> BroadcastMulOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast("a").PartialSum("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().PartialSum("a").Broadcast("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kBroadcastMulConf, BroadcastMulOp);

}  // namespace oneflow
