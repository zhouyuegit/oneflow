#include "oneflow/core/operator/sparse_cross_entropy_op.h"

namespace oneflow {

namespace {

class CrossEntropyOpModelSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CrossEntropyOpModelSplitSignature);
  ~CrossEntropyOpModelSplitSignature() override = default;

  CrossEntropyOpModelSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(1) -> P"; }

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
    if (SbpInferHint4BnInOp(ibn).sbp_parallel().split_parallel().axis() != 1) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["prediction"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["label"].mutable_broadcast_parallel();
    // (*bn2sbp)["out"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["out"].mutable_partial_sum_parallel();
  }
};

class CrossEntropyOpDataSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CrossEntropyOpDataSplitSignature);
  ~CrossEntropyOpDataSplitSignature() override = default;

  CrossEntropyOpDataSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> S(0)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    if (!SbpInferHint4BnInOp("prediction").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("prediction").sbp_parallel().split_parallel().axis() != 0) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (!SbpInferHint4BnInOp("label").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("label").sbp_parallel().split_parallel().axis() != 0) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["prediction"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["label"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(0);
  }
};

}


void SparseCrossEntropyOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("out");
}

const PbMessage& SparseCrossEntropyOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_conf();
}

void SparseCrossEntropyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK(IsFloatingDataType(pred_blob_desc->data_type()));
  CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
  CHECK_EQ(pred_blob_desc->has_dim0_valid_num_field(), label_blob_desc->has_dim0_valid_num_field());
  CHECK_EQ(pred_blob_desc->has_dim0_inner_shape(), label_blob_desc->has_dim0_inner_shape());
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(pred_blob_desc->dim0_inner_shape().At(0), 1);
    CHECK_EQ(pred_blob_desc->dim0_inner_shape(), label_blob_desc->dim0_inner_shape());
  }
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  const int64_t num_out_axes = pred_blob_desc->shape().NumAxes() - 1;
  CHECK_GE(label_blob_desc->shape().NumAxes(), num_out_axes);
  CHECK_EQ(label_blob_desc->shape().Count(num_out_axes), 1);
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ(pred_blob_desc->shape().At(i), label_blob_desc->shape().At(i));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *pred_blob_desc;
  out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
      pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
}

void SparseCrossEntropyOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new CrossEntropyOpModelSplitSignature(this));
  op_parallel_signatures->emplace_back(new CrossEntropyOpDataSplitSignature(this));
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);

}  // namespace oneflow
