#include "oneflow/core/operator/l2_normalize_op.h"

namespace oneflow {

namespace {

class L2NormOpBroadcastSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormOpBroadcastSignature);
  ~L2NormOpBroadcastSignature() override = default;

  L2NormOpBroadcastSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": B -> B"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    if (!SbpInferHint4BnInOp(ibn).sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    CHECK(SbpInferHint4BnInOp("in").is_model_broadcast());
    (*bn2sbp)["in"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_broadcast_parallel();
  }
};

class L2NormOpDataSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormOpDataSplitSignature);
  ~L2NormOpDataSplitSignature() override = default;

  L2NormOpDataSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> S(0)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    if (!SbpInferHint4BnInOp(ibn).sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp(ibn).sbp_parallel().split_parallel().axis() != 0) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(0);
  }
};

}  // namespace

void L2NormalizeOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("square_x_sum");
}

const PbMessage& L2NormalizeOp::GetCustomizedConf() const { return op_conf().l2_normalize_conf(); }

void L2NormalizeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const L2NormalizeOpConf& conf = op_conf().l2_normalize_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int32_t axis_num = in_blob_desc->shape().NumAxes();
  int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + axis_num;
  CHECK_GE(axis, 0);
  CHECK_LT(axis, axis_num);
  CHECK_GT(conf.epsilon(), 0);
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc* square_x_sum_blob_desc = GetBlobDesc4BnInOp("square_x_sum");
  *square_x_sum_blob_desc = *in_blob_desc;
  square_x_sum_blob_desc->mut_shape().Set(axis, 1);
}

void L2NormalizeOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new L2NormOpBroadcastSignature(this));
  op_parallel_signatures->emplace_back(new L2NormOpDataSplitSignature(this));
}

REGISTER_OP(OperatorConf::kL2NormalizeConf, L2NormalizeOp);

}  // namespace oneflow
