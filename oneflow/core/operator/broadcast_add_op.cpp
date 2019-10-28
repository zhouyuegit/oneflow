#include "oneflow/core/operator/broadcast_add_op.h"

namespace oneflow {

const PbMessage& BroadcastAddOp::GetCustomizedConf() const {
  return op_conf().broadcast_add_conf();
}

Maybe<void> BroadcastAddOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("a").PartialSum("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kBroadcastAddConf, BroadcastAddOp);

}  // namespace oneflow
