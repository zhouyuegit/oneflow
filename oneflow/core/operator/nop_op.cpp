#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NopOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NopOp);
  NopOp() = default;
  ~NopOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().nop_conf(); }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void NopOp::InitFromOpConf() {
  if (op_conf().has_nop_conf()) { EnrollInputBn("tick", false); }
}

Maybe<void> NopOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  return Maybe<void>::Ok();
}

Maybe<void> NopOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return Maybe<void>::Ok();
}

Maybe<void> NopOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast(input_bns()).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kNopConf, NopOp);

}  // namespace oneflow