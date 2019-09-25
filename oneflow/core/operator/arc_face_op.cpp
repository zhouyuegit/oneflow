#include "oneflow/core/operator/arc_face_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace {

class ArcFaceOpModelSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArcFaceOpModelSplitSignature);
  ~ArcFaceOpModelSplitSignature() override = default;

  ArcFaceOpModelSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(1), B -> S(1)"; }

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
    if (!SbpInferHint4BnInOp("label").sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["label"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(1);
  }
};

}  // namespace

void ArcFaceOp::InitFromOpConf() {
  CHECK(op_conf().has_arc_face_conf());
  EnrollInputBn("in");
  EnrollInputBn("label", false);
  EnrollDataTmpBn("sin_theta_data");
  EnrollOutputBn("out");
}

const PbMessage& ArcFaceOp::GetCustomizedConf() const { return op_conf().arc_face_conf(); }

void ArcFaceOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const BlobDesc* label = GetBlobDesc4BnInOp("label");
  CHECK_GT(label->shape().NumAxes(), 0);
  CHECK(IsIntegralDataType(label->data_type()));
  CHECK_EQ(label->shape().At(0), in->shape().At(0));

  BlobDesc* sin_theta_data = GetBlobDesc4BnInOp("sin_theta_data");
  sin_theta_data->set_data_type(in->data_type());
  sin_theta_data->mut_shape() = label->shape();

  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

void ArcFaceOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ArcFaceOpModelSplitSignature(this));
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kArcFaceConf, ArcFaceOp);

}  // namespace oneflow
