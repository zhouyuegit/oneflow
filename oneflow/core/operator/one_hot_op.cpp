#include "oneflow/core/operator/one_hot_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace {

class OneHotOpModelSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneHotOpModelSplitSignature);
  ~OneHotOpModelSplitSignature() override = default;

  OneHotOpModelSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": B -> S(1)"; }

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
    (*bn2sbp)["indices"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(1);
  }
};
}


void OneHotOp::InitFromOpConf() {
  CHECK(op_conf().has_one_hot_conf());
  EnrollInputBn("indices", false);
  EnrollOutputBn("out", false);
}

const PbMessage& OneHotOp::GetCustomizedConf() const { return op_conf().one_hot_conf(); }

void OneHotOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const OneHotOpConf& conf = op_conf().one_hot_conf();
  int64_t depth = conf.depth();
  const DataType data_type =
      conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  CHECK_GT(depth, 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *indices;
  out->set_data_type(data_type);
  std::vector<int64_t> dim_vec = indices->shape().dim_vec();

  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(depth, parallel_ctx->parallel_num());
    depth = splitter.At(parallel_ctx->parallel_id()).size();
  }
  dim_vec.push_back(depth);
  out->mut_shape() = Shape(dim_vec);
}

void OneHotOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new OneHotOpModelSplitSignature(this));
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kOneHotConf, OneHotOp);

}  // namespace oneflow
