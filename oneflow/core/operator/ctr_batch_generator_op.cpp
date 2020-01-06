#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class CtrBatchGeneratorOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrBatchGeneratorOp);
  CtrBatchGeneratorOp() = default;
  ~CtrBatchGeneratorOp() override = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

void CtrBatchGeneratorOp::InitFromOpConf() {
  CHECK(op_conf().has_ctr_batch_generator_conf());
  const CtrBatchGeneratorOpConf& conf = op_conf().ctr_batch_generator_conf();
  if (conf.has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("label", false);
  EnrollRepeatedOutputBn("feature_id", false);
  EnrollRepeatedOutputBn("feature_slot", false);
}

const PbMessage& CtrBatchGeneratorOp::GetCustomizedConf() const {
  return op_conf().ctr_batch_generator_conf();
}

Maybe<void> CtrBatchGeneratorOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  CHECK_EQ(parallel_ctx->parallel_num(), 1);
  const CtrBatchGeneratorOpConf& conf = op_conf().ctr_batch_generator_conf();
  const int64_t batch_size = conf.batch_size();
  BlobDesc* label = GetBlobDesc4BnInOp("label");
  label->mut_shape() = Shape({batch_size});
  label->set_data_type(DataType::kInt8);
  FOR_RANGE(int64_t, i, 0, conf.num_partition()) {
    BlobDesc* feature_id = GetBlobDesc4BnInOp(GenRepeatedBn("feature_id", i));
    feature_id->mut_shape() = Shape({batch_size * conf.max_num_feature()});
    feature_id->set_is_dynamic(true);
    feature_id->set_data_type(DataType::kInt32);
    BlobDesc* feature_slot = GetBlobDesc4BnInOp(GenRepeatedBn("feature_slot", i));
    *feature_slot = *feature_id;
  }
  return Maybe<void>::Ok();
}

Maybe<void> CtrBatchGeneratorOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->set_value(0); }
  return Maybe<void>::Ok();
}

Maybe<void> CtrBatchGeneratorOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kCtrBatchGeneratorConf, CtrBatchGeneratorOp);

}  // namespace oneflow
