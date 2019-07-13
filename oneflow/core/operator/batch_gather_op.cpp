#include "oneflow/core/operator/batch_gather_op.h"

namespace oneflow {

namespace {

class BatchGatherModelSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGatherModelSplitSignature);
  ~BatchGatherModelSplitSignature() override = default;

  BatchGatherModelSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (S(1), B) -> P"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp("in").parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
          SbpInferHint4BnInOp("in").parallel_num());
    }
    if (!SbpInferHint4BnInOp("in").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("in").sbp_parallel().split_parallel().axis() != 1) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (!SbpInferHint4BnInOp("indices").sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["indices"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_partial_sum_parallel();
  }
};

}  // namespace

void BatchGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_gather_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
  EnrollOutputBn("out");
}

const PbMessage& BatchGatherOp::GetCustomizedConf() const { return op_conf().batch_gather_conf(); }

void BatchGatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_GT(indices->shape().NumAxes(), 0);
  CHECK(IsIntegralDataType(indices->data_type()));
  const std::vector<int64_t>& in_dim_vec = in->shape().dim_vec();
  const std::vector<int64_t>& indices_dim_vec = indices->shape().dim_vec();
  CHECK_LE(indices_dim_vec.size(), in_dim_vec.size());
  FOR_RANGE(int64_t, i, 0, indices_dim_vec.size() - 1) {
    CHECK_EQ(indices_dim_vec.at(i), in_dim_vec.at(i));
  }
  // out
  std::vector<int64_t> out_dim_vec(indices_dim_vec);
  out_dim_vec.insert(out_dim_vec.end(), in_dim_vec.cbegin() + indices_dim_vec.size(),
                     in_dim_vec.cend());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(out_dim_vec);
}

void BatchGatherOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  // op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
  op_parallel_signatures->emplace_back(new BatchGatherModelSplitSignature(this));
}

REGISTER_OP(OperatorConf::kBatchGatherConf, BatchGatherOp);

}  // namespace oneflow
