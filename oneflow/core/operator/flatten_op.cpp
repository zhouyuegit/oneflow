#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FlattenOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FlattenOp);
  FlattenOp() = default;
  ~FlattenOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override;
};

void FlattenOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

const PbMessage& FlattenOp::GetCustomizedConf() const { return op_conf().flatten_conf(); }

Maybe<void> FlattenOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int64_t num_in_axes = in->shape().NumAxes();
  const FlattenOpConf& conf = op_conf().flatten_conf();
  const int64_t begin_axis = conf.begin_axis();
  const int64_t end_axis = conf.end_axis();
  CHECK_GE(begin_axis, 0);
  CHECK_LT(begin_axis, num_in_axes);
  CHECK_GE(end_axis, 0);
  CHECK_LT(end_axis, num_in_axes);
  CHECK_LE(begin_axis, end_axis);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  std::vector<int64_t> out_dim_vec;
  FOR_RANGE(int64_t, i, 0, begin_axis) { out_dim_vec.push_back(in->shape().At(i)); }
  if (begin_axis < end_axis) { out_dim_vec.push_back(in->shape().Count(begin_axis, end_axis)); }
  FOR_RANGE(int64_t, i, end_axis, num_in_axes) { out_dim_vec.push_back(in->shape().At(i)); }
  out->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> FlattenOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {}

REGISTER_OP(OperatorConf::kFlattenConf, FlattenOp);

}  // namespace oneflow
