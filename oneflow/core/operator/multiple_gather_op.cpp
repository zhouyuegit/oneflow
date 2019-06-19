#include "oneflow/core/operator/multiple_gather_op.h"

namespace oneflow {

namespace {

class MultipleGather_OpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultipleGather_OpParallelSignature);
  ~MultipleGather_OpParallelSignature() override = default;

  MultipleGather_OpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S->S or B->B"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"].mutable_broadcast_parallel();
		const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    FOR_RANGE(int32_t, i, 0, op().output_bns().size()) {
      //const auto& sbp_parallel = SbpInferHint4BnInOp(GenRepeatedBn("indices", i)).sbp_parallel();
			const SbpInferHint& sbp_infer_hint = SbpInferHint4BnInOp(GenRepeatedBn("indices", i));
      const SbpParallel& sbp_parallel = sbp_infer_hint.sbp_parallel();
      (*bn2sbp)[GenRepeatedBn("indices", i)] = sbp_parallel;
      (*bn2sbp)[GenRepeatedBn("out", i)] = sbp_parallel;
    }
  }
};

}  // namespace

void MultipleGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_multiple_gather_conf());
  EnrollRepeatedInputBn("indices", false);
  EnrollInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& MultipleGatherOp::GetCustomizedConf() const {
  return op_conf().multiple_gather_conf();
}

void MultipleGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const MultipleGatherOpConf& conf = op_conf().multiple_gather_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  CHECK_GT(conf.indices().size(), 0);
  CHECK_EQ(conf.indices().size(), conf.out().size());
  FOR_RANGE(int32_t, i, 0, conf.indices().size()) {
    const BlobDesc* indices = GetBlobDesc4BnInOp(GenRepeatedBn("indices", i));
    CHECK(IsIntegralDataType(indices->data_type()));
    CHECK_GT(indices->shape().NumAxes(), 0);
    BlobDesc* out = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    *out = *in;
    std::vector<int64_t> dim_vec;
    dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                   indices->shape().dim_vec().cend());
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + 1, in->shape().dim_vec().end());
    out->mut_shape() = Shape(dim_vec);
  }
}

int32_t MultipleGatherOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  UNIMPLEMENTED();
  return -1;
}

void MultipleGatherOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new MultipleGather_OpParallelSignature(this));
}

void MultipleGatherOp::InferIsModelBlob4OutputBlobs(
    std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, output_bns().size()) {
    bool is_model_blob =  *IsModelBlob4BnInOp(GenRepeatedBn("indices", i)); 
    *IsModelBlob4BnInOp(GenRepeatedBn("out", i)) = *IsModelBlob4BnInOp(GenRepeatedBn("indices", i)); 
  }
}
REGISTER_OP(OperatorConf::kMultipleGatherConf, MultipleGatherOp);

}  // namespace oneflow
