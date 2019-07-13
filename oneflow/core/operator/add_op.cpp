#include "oneflow/core/operator/add_op.h"

namespace oneflow {

namespace {

class AddAnyAxisSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddAnyAxisSplitSignature);
  ~AddAnyAxisSplitSignature() override = default;

  AddAnyAxisSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S -> S"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn_0 = op().input_bns().Get(0);
    for (size_t i = 0; i < op().input_bns().size(); ++i) {
      const auto& ibn = op().input_bns().Get(i);
      if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
        return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                   SbpInferHint4BnInOp(ibn).parallel_num());
      }
      if (!SbpInferHint4BnInOp(ibn).sbp_parallel().has_split_parallel()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
      if (SbpInferHint4BnInOp(ibn_0).sbp_parallel().split_parallel().axis()
          != SbpInferHint4BnInOp(ibn).sbp_parallel().split_parallel().axis()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    CHECK(SbpInferHint4BnInOp(op().input_bns().Get(0)).sbp_parallel().has_split_parallel());
    const auto& axis = SbpInferHint4BnInOp(
                           op().input_bns().Get(0)).sbp_parallel().split_parallel().axis();
    for (size_t i = 0; i < op().input_bns().size(); ++i) {
      (*bn2sbp)[op().input_bns().Get(i)].mutable_split_parallel()->set_axis(axis);
    }
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(axis);
  }
};

} // namespace

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }
void AddOp::VirtualFixInDiffBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (!Global<JobDesc>::Get()->enable_blob_mem_sharing()) { return; }
  int64_t blob_mem_id = oneflow_cast<int64_t>(NewUniqueId());
  FOR_RANGE(size_t, i, 0, input_diff_bns().size()) {
    GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_blob_mem_id(blob_mem_id);
  }
}

void AddOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new AddAnyAxisSplitSignature(this));
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
