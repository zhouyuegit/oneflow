#include "oneflow/core/operator/softmax_reduce_max_stage1_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace {

class ReduceMaxOpBroadcastSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMaxOpBroadcastSignature);
  ~ReduceMaxOpBroadcastSignature() override = default;

  ReduceMaxOpBroadcastSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": B -> B"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    if (!SbpInferHint4BnInOp("in").sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (!SbpInferHint4BnInOp("max_count").sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    // CHECK(SbpInferHint4BnInOp("in").is_model_broadcast());
    (*bn2sbp)["in"].mutable_broadcast_parallel();
    (*bn2sbp)["max_count"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_broadcast_parallel();
  }
};

class ReduceMaxOpDataSplitSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMaxOpDataSplitSignature);
  ~ReduceMaxOpDataSplitSignature() override = default;

  ReduceMaxOpDataSplitSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> S(0)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    if (!SbpInferHint4BnInOp("in").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("in").sbp_parallel().split_parallel().axis() != 0) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (!SbpInferHint4BnInOp("max_count").sbp_parallel().has_split_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (SbpInferHint4BnInOp("max_count").sbp_parallel().split_parallel().axis() != 0) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    // CHECK(SbpInferHint4BnInOp("in").is_model_broadcast());
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["max_count"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(0);
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

void SoftmaxReduceMaxStage1Op::InitFromOpConf() {
  CHECK(op_conf().has_softmax_reduce_max_stage1_conf());
  EnrollInputBn("in");
  EnrollInputBn("max_count");
  EnrollOutputBn("out");
  EnrollDataTmpBn("mask");
  EnrollDataTmpBn("data_tmp");
  EnrollDataTmpBn("data_tmp_int");
  EnrollBwBufBn("max_count_with_mask");
  EnrollBwBufBn("global_max_count");  // TODO(shiyuan) ?
}

const PbMessage& SoftmaxReduceMaxStage1Op::GetCustomizedConf() const { return op_conf().softmax_reduce_max_stage1_conf(); }

void SoftmaxReduceMaxStage1Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const SoftmaxReduceMaxStage1OpConf& conf = op_conf().softmax_reduce_max_stage1_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("data_tmp") = *in_blob;
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
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  out_blob->mut_shape() = Shape(out_dim_vec);

  BlobDesc* mask_blob = GetBlobDesc4BnInOp("mask");
  mask_blob->set_data_type(DataType::kInt32);
  mask_blob->mut_shape() = in_blob->shape();

  BlobDesc* data_tmp_int_blob = GetBlobDesc4BnInOp("data_tmp_int");
  data_tmp_int_blob->set_data_type(DataType::kInt32);
  data_tmp_int_blob->mut_shape() = in_blob->shape();

  // TODO(shiyuan) ?
  // *GetBlobDesc4BnInOp("global_max_count") = *GetBlobDesc4BnInOp("out");
}


void SoftmaxReduceMaxStage1Op::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* max_count_blob_desc = GetBlobDesc4BnInOp("max_count");
  *GetBlobDesc4BnInOp("max_count_with_mask") = *max_count_blob_desc;
  *GetBlobDesc4BnInOp("global_max_count") = *out_blob_desc;

  BlobDesc* global_max_count_blob_desc = GetBlobDesc4BnInOp("global_max_count");
  global_max_count_blob_desc->set_data_type(DataType::kInt32);
  global_max_count_blob_desc->mut_shape() = out_blob_desc->shape();
}

void SoftmaxReduceMaxStage1Op::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  const SoftmaxReduceMaxStage1OpConf& conf = op_conf().softmax_reduce_max_stage1_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> kept_dims;
  if (conf.axis().empty()) {
    kept_dims.resize(in_blob->shape().NumAxes());
    std::fill(kept_dims.begin(), kept_dims.end(), 1);
  } else {
    const PbRf<int32_t>& axis_repeated = op_conf().softmax_reduce_max_stage1_conf().axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    kept_dims = KeepDims(in_blob->shape().dim_vec(),
                         ShiftAxisIfNegative(axis_vec, in_blob->shape().NumAxes()));
  }
  Shape(kept_dims).ToProto(kernel_conf->mutable_reduce_sum_conf()->mutable_kept_dims_shape());
}

void SoftmaxReduceMaxStage1Op::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ReduceMaxOpBroadcastSignature(this));
  op_parallel_signatures->emplace_back(new ReduceMaxOpDataSplitSignature(this));
}

REGISTER_OP(OperatorConf::kSoftmaxReduceMaxStage1Conf, SoftmaxReduceMaxStage1Op);

}  // namespace oneflow
