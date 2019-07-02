#include "oneflow/core/operator/softmax_reduce_max_stage0_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace {

class ReduceMaxOpModelSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMaxOpModelSplitSignature);
  ~ReduceMaxOpModelSplitSignature() override = default;

  ReduceMaxOpModelSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(1) -> S(1)"; }

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
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["max"].mutable_split_parallel()->set_axis(1);
    (*bn2sbp)["max_count"].mutable_split_parallel()->set_axis(1);
  }
};

std::vector<int64_t> KeepDims(const std::vector<int64_t> dim_vec,
                              const std::vector<int64_t> axis_vec) {
  std::vector<int64_t> ret = dim_vec;
  for (const auto& axis : axis_vec) { ret[axis] = 1; }
  return ret;
}

std::vector<int64_t> DropDims(const std::vector<int64_t> dim_vec,
                              const std::vector<int64_t> axis_vec) {
  std::vector<int64_t> ret;
  std::vector<int32_t> dim2is_reduced(dim_vec.size());
  for (const auto& axis : axis_vec) { dim2is_reduced[axis] = 1; }
  FOR_RANGE(int64_t, i, 0, dim_vec.size()) {
    if (dim2is_reduced[i] != 1) { ret.push_back(dim_vec[i]); }
  }
  if (ret.empty()) { ret.push_back(1); }
  return ret;
}

std::vector<int64_t> ShiftAxisIfNegative(std::vector<int64_t> axis_vec, const int64_t num_axes) {
  FOR_RANGE(size_t, i, 0, axis_vec.size()) {
    if (axis_vec[i] < 0) { axis_vec[i] += num_axes; }
    CHECK_LT(axis_vec[i], num_axes);
    CHECK_GE(axis_vec[i], 0);
  }
  return axis_vec;
}

}  // namespace

void SoftmaxReduceMaxStage0Op::InitFromOpConf() {
  CHECK(op_conf().has_softmax_reduce_max_stage0_conf());
  EnrollInputBn("in");
  EnrollOutputBn("max");
  EnrollOutputBn("max_count");
  EnrollFwBufBn("fw_tmp");
  EnrollFwBufBn("fw_tmp_int");
  EnrollDataTmpBn("mask");
}

const PbMessage& SoftmaxReduceMaxStage0Op::GetCustomizedConf() const { return op_conf().softmax_reduce_max_stage0_conf(); }

void SoftmaxReduceMaxStage0Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const SoftmaxReduceMaxStage0OpConf& conf = op_conf().softmax_reduce_max_stage0_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;

  std::vector<int64_t> out_dim_vec;
  if (conf.axis().empty()) {
    if (conf.keep_dims() == true) {
      out_dim_vec.resize(in_blob->shape().NumAxes());
      std::fill(out_dim_vec.begin(), out_dim_vec.end(), 1);
    } else {
      out_dim_vec = {1};
    }
  } else {
    const PbRf<int32_t>& axis_repeated = conf.axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    axis_vec = ShiftAxisIfNegative(axis_vec, in_blob->shape().NumAxes());
    std::sort(axis_vec.begin(), axis_vec.end());
    CHECK(std::unique(axis_vec.begin(), axis_vec.end()) == axis_vec.end())
        << "duplicate found in axis";
    if (conf.keep_dims() == true) {
      out_dim_vec = KeepDims(in_blob->shape().dim_vec(), axis_vec);
    } else {
      out_dim_vec = DropDims(in_blob->shape().dim_vec(), axis_vec);
    }
  }
  CHECK(!out_dim_vec.empty());
  BlobDesc* max_blob = GetBlobDesc4BnInOp("max");
  max_blob->set_data_type(in_blob->data_type());
  max_blob->mut_shape() = Shape(out_dim_vec);

  BlobDesc* max_count_blob = GetBlobDesc4BnInOp("max_count");
  max_count_blob->set_data_type(DataType::kInt32);
  max_count_blob->mut_shape() = Shape(out_dim_vec);

  BlobDesc* mask_blob = GetBlobDesc4BnInOp("mask");
  mask_blob->set_data_type(DataType::kInt32);
  mask_blob->mut_shape() = in_blob->shape();

  BlobDesc* fw_tmp_int_blob = GetBlobDesc4BnInOp("fw_tmp_int");
  fw_tmp_int_blob->set_data_type(DataType::kInt32);
  fw_tmp_int_blob->mut_shape() = in_blob->shape();
}

void SoftmaxReduceMaxStage0Op::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  const SoftmaxReduceMaxStage0OpConf& conf = op_conf().softmax_reduce_max_stage0_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> kept_dims;
  if (conf.axis().empty()) {
    kept_dims.resize(in_blob->shape().NumAxes());
    std::fill(kept_dims.begin(), kept_dims.end(), 1);
  } else {
    const PbRf<int32_t>& axis_repeated = op_conf().softmax_reduce_max_stage0_conf().axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    kept_dims = KeepDims(in_blob->shape().dim_vec(),
                         ShiftAxisIfNegative(axis_vec, in_blob->shape().NumAxes()));
  }
  Shape(kept_dims).ToProto(kernel_conf->mutable_reduce_sum_conf()->mutable_kept_dims_shape());
}


void SoftmaxReduceMaxStage0Op::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ReduceMaxOpModelSplitSignature(this));
}

REGISTER_OP(OperatorConf::kSoftmaxReduceMaxStage0Conf, SoftmaxReduceMaxStage0Op);

}  // namespace oneflow
