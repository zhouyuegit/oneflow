#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class DecodeOneRecOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOneRecOp);
  DecodeOneRecOp() = default;
  ~DecodeOneRecOp() override = default;

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

void DecodeOneRecOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_onerec_conf());
  const DecodeOneRecOpConf& conf = op_conf().decode_onerec_conf();
  if (conf.has_tick()) { EnrollInputBn("tick", false); }
  EnrollRepeatedOutputBn("out", false);
}

const PbMessage& DecodeOneRecOp::GetCustomizedConf() const {
  return op_conf().decode_onerec_conf();
}

Maybe<void> DecodeOneRecOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const DecodeOneRecOpConf& conf = op_conf().decode_onerec_conf();
  const int64_t batch_size = conf.batch_size();
  CHECK_EQ(batch_size % parallel_ctx->parallel_num(), 0);
  const int64_t device_batch_size = batch_size / parallel_ctx->parallel_num();
  CHECK_EQ(output_bns().size(), conf.field_size());
  FOR_RANGE(int64_t, i, 0, output_bns().size()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    const DecodeOneRecFieldConf& field_conf = conf.field().Get(i);
    const int64_t num_output_axes = 1 + field_conf.output_shape().dim_size();
    std::vector<int64_t> dim_vec(num_output_axes);
    dim_vec[0] = device_batch_size;
    FOR_RANGE(int64_t, j, 1, num_output_axes) { dim_vec[j] = field_conf.output_shape().dim(j - 1); }
    out_blob_desc->mut_shape() = Shape(dim_vec);
    out_blob_desc->set_data_type(field_conf.output_data_type());
  }
  return Maybe<void>::Ok();
}

Maybe<void> DecodeOneRecOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->set_value(0); }
  return Maybe<void>::Ok();
}

Maybe<void> DecodeOneRecOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignatureBuilder().Broadcast(input_bns()).Split(output_bns(), 0).Build(sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kDecodeOnerecConf, DecodeOneRecOp);

}  // namespace oneflow
