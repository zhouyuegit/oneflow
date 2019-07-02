#include "oneflow/core/operator/multiply_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

namespace {

class MultiplySplit1Signature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiplySplit1Signature);
  ~MultiplySplit1Signature() override = default;

  MultiplySplit1Signature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(1) -> S(1)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp("in_0").parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
          SbpInferHint4BnInOp("in_0").parallel_num());
    }
    if (!SbpInferHint4BnInOp("in_0").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("in_0").sbp_parallel().split_parallel().axis() != 1) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (!SbpInferHint4BnInOp("in_1").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("in_1").sbp_parallel().split_parallel().axis() != 1) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in_0"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["in_1"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(1);
  }
};

} // namespace

void MultiplyOp::InitFromOpConf() {
  CHECK(op_conf().has_multiply_conf());
  EnrollInputBn("in_0");
  EnrollInputBn("in_1", false);
  EnrollOutputBn("out");
}

const PbMessage& MultiplyOp::GetCustomizedConf() const { return op_conf().multiply_conf(); }

void MultiplyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp("in_0");
  BlobDesc* in_1_blob_desc = GetBlobDesc4BnInOp("in_1");
  CHECK_EQ(in_0_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(in_0_blob_desc->shape(), in_1_blob_desc->shape());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
}

void MultiplyOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new MultiplySplit1Signature(this));
}

REGISTER_OP(OperatorConf::kMultiplyConf, MultiplyOp);

}  // namespace oneflow
