#include "oneflow/core/operator/operator.h"

namespace oneflow {

class IndexedSlicesLazyAdamOptimizerOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesLazyAdamOptimizerOp);
  IndexedSlicesLazyAdamOptimizerOp() = default;
  ~IndexedSlicesLazyAdamOptimizerOp() override = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void IndexedSlicesLazyAdamOptimizerOp::InitFromOpConf() {
  const auto& conf = op_conf().indexed_slices_lazy_adam_optimizer_conf();
  CHECK_GE(conf.beta1(), 0);
  CHECK_LT(conf.beta1(), 1);
  CHECK_GE(conf.beta2(), 0);
  CHECK_LT(conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  EnrollInputBn("model_diff_indices", false);
  EnrollInputBn("model_diff_values", false);
  EnrollInputBn("total_instance_num_diff", false);
  EnrollInputBn("model", false)->set_is_mutable(true);
  EnrollInputBn("train_step", false);
  EnrollInputBn("learning_rate", false);
}

Maybe<void> IndexedSlicesLazyAdamOptimizerOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesLazyAdamOptimizerOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("m", 0)
      .Split("v", 0)
      .Split("model", 0)
      .Broadcast("model_diff_indices")
      .Broadcast("model_diff_values")
      .Broadcast("total_instance_num_diff")
      .Broadcast("train_step")
      .Broadcast("learning_rate")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

const PbMessage& IndexedSlicesLazyAdamOptimizerOp::GetCustomizedConf() const {
  return op_conf().indexed_slices_lazy_adam_optimizer_conf();
}

REGISTER_OP(OperatorConf::kIndexedSlicesLazyAdamOptimizerConf, IndexedSlicesLazyAdamOptimizerOp);

}  // namespace oneflow
