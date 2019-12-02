#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OnesLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OnesLikeOp);
  OnesLikeOp() = default;
  ~OnesLikeOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;
  bool IsAllOutputConst() const override { return true; }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void OnesLikeOp::InitFromOpConf() {
  EnrollInputBn("like", false);
  EnrollOutputBn("out", false);
}

const PbMessage& OnesLikeOp::GetCustomizedConf() const { return op_conf().ones_like_conf(); }

Maybe<void> OnesLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* like = GetBlobDesc4BnInOp("like");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  const OnesLikeOpConf& conf = op_conf().ones_like_conf();
  const DataType& data_type = conf.has_data_type() ? conf.data_type() : like->data_type();
  *out = *like;
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

Maybe<void> OnesLikeOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("like");
  return Maybe<void>::Ok();
}

Maybe<void> OnesLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("like"))->shape().NumAxes();
  SbpSignatureBuilder()
      .Split("like", 0)
      .Split("out", 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
  SbpSignatureBuilder().PartialSum("like").Broadcast("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kOnesLikeConf, OnesLikeOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kOnesLikeConf, 1);

}  // namespace oneflow
