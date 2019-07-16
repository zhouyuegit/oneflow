#include "oneflow/core/operator/expand_dims_op.h"

namespace oneflow {

namespace {

class ExpandDimsOpBroadcastSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpandDimsOpBroadcastSignature);
  ~ExpandDimsOpBroadcastSignature() override = default;

  ExpandDimsOpBroadcastSignature(const Operator* op) : OpParallelSignature(op) {}

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
    (*bn2sbp)["in"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ExpandDimsOp::InitFromOpConf() {
  CHECK(op_conf().has_expand_dims_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ExpandDimsOp::GetCustomizedConf() const { return op_conf().expand_dims_conf(); }

void ExpandDimsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  std::vector<int64_t> dim_vec = in_blob_desc->shape().dim_vec();
  int32_t dim = op_conf().expand_dims_conf().dim();
  CHECK_GE(dim, -static_cast<int32_t>(dim_vec.size()) - 1);
  CHECK_LE(dim, dim_vec.size());
  std::vector<int64_t>::iterator it;
  if (dim >= 0) {
    it = dim_vec.begin() + dim;
  } else {
    it = dim_vec.end() + 1 + dim;
  }
  dim_vec.insert(it, 1);
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
}

void ExpandDimsOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ExpandDimsOpBroadcastSignature(this));
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kExpandDimsConf, ExpandDimsOp);

}  // namespace oneflow
