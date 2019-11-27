#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ScatterAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScatterAddOp);
  ScatterAddOp() = default;
  ~ScatterAddOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void ScatterAddOp::InitFromOpConf() {
  CHECK(op_conf().has_scatter_add_conf());
  EnrollInputBn("ref")->set_is_mutable(true);
  EnrollInputBn("indices");
  EnrollInputBn("updates");
}

const PbMessage& ScatterAddOp::GetCustomizedConf() const { return op_conf().scatter_add_conf(); }

Maybe<void> ScatterAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* ref = GetBlobDesc4BnInOp("ref");
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
  CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
  CHECK_EQ_OR_RETURN(ref->data_type(), updates->data_type());
  const int64_t num_indices_axes = indices->shape().NumAxes();
  const int64_t num_updates_axes = updates->shape().NumAxes();
  CHECK_LE_OR_RETURN(num_indices_axes, num_updates_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ_OR_RETURN(indices->shape().At(i), updates->shape().At(i));
  }
  const int64_t num_ref_axes = ref->shape().NumAxes();
  CHECK_EQ_OR_RETURN(num_updates_axes - num_indices_axes + 1, num_ref_axes);
  FOR_RANGE(int64_t, i, 1, num_ref_axes) {
    CHECK_EQ_OR_RETURN(ref->shape().At(i), updates->shape().At(num_indices_axes + i - 1));
  }
  return Maybe<void>::Ok();
}

Maybe<void> ScatterAddOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_indices_axes = JUST(LogicalBlobDesc4Ibn("indices"))->shape().NumAxes();
  const int64_t num_ref_axes = JUST(LogicalBlobDesc4Ibn("ref"))->shape().NumAxes();
  SbpSignatureBuilder().Broadcast("indices").Broadcast("updates").Split("ref", 0).Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  FOR_RANGE(int64_t, i, 1, num_ref_axes) {
    SbpSignatureBuilder()
        .Broadcast("indices")
        .Split("updates", num_indices_axes + i - 1)
        .Split("ref", i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kScatterAddConf, ScatterAddOp);

}  // namespace oneflow
