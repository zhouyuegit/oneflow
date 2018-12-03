#include "oneflow/core/operator/pad_op.h"

namespace oneflow {

void PadOp::InitFromOpConf() {
  CHECK(op_conf().has_pad_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollDataTmpBn("padding_left_bound");
  EnrollDataTmpBn("padding_right_bound");
  EnrollDataTmpBn("inshape_count");
  EnrollDataTmpBn("outshape_count");
}

const PbMessage& PadOp::GetCustomizedConf() const { return op_conf().pad_conf(); }

void PadOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_shape = in_blob_desc->shape();
  int64_t num_axes = in_blob_desc->shape().NumAxes();
  CHECK_GE(num_axes, 3);
  CHECK_LE(num_axes, 5);
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  // out
  const std::string data_format = op_conf().pad_conf().data_format();
  int64_t dims = in_blob_desc->shape().NumAxes() - 2;
  std::vector<int64_t> in = {GetInDim(in_shape, data_format, 0, dims),
                             GetInDim(in_shape, data_format, 1, dims),
                             GetInDim(in_shape, data_format, 2, dims)};
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  int64_t in_c = 0;
  if (data_format== "channels_first") {
    in_c = in_shape.At(1);
  } else if (data_format == "channels_last") {
    in_c = in_shape.At(in_shape.NumAxes() - 1);
  } else {
    UNIMPLEMENTED();
  }
  std::vector<int32_t> padding_after =
      GetPaddingsVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("padding_after"), num_axes - 2);
  std::vector<int32_t> padding_before =
      GetPaddingsVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("padding_before"), num_axes - 2);
  CHECK_EQ(padding_before.size(), num_axes - 2);
  CHECK_EQ(padding_after.size(), num_axes - 2);
  out_blob_desc->mut_shape() = GetOutShape(in_shape.At(0), in_c, dims, data_format, 
                                           in, padding_before, padding_after);
  // tmp blobs
  BlobDesc* padding_left_bound_blob_desc = GetBlobDesc4BnInOp("padding_left_bound");
  padding_left_bound_blob_desc->mut_shape() = Shape({num_axes});
  padding_left_bound_blob_desc->set_data_type(DataType::kInt32);

  BlobDesc* padding_right_bound_blob_desc = GetBlobDesc4BnInOp("padding_right_bound");
  padding_right_bound_blob_desc->mut_shape() = Shape({num_axes});
  padding_right_bound_blob_desc->set_data_type(DataType::kInt32);

  BlobDesc* inshape_count_blob_desc = GetBlobDesc4BnInOp("inshape_count");
  inshape_count_blob_desc->mut_shape() = Shape({num_axes});
  inshape_count_blob_desc->set_data_type(DataType::kInt32);

  BlobDesc* outshape_count_blob_desc = GetBlobDesc4BnInOp("outshape_count");
  outshape_count_blob_desc->mut_shape() = Shape({num_axes});
  outshape_count_blob_desc->set_data_type(DataType::kInt32);

}

Shape PadOp::GetOutShape(int64_t in_n, int64_t in_c, int64_t dims, 
                         std::string data_format, const std::vector<int64_t>& in,
                         const std::vector<int32_t>& padding_before,
                         const std::vector<int32_t>& padding_after) const {
  std::vector<int64_t> out_shape;
  if (dims == 1) {
    out_shape = {in.at(2) + padding_after.at(0) + padding_before.at(0)};
  } else if (dims == 2) {
    out_shape = {in.at(1) + padding_after.at(0) + padding_before.at(0), 
                 in.at(2) + padding_after.at(1) + padding_before.at(1)};
  } else if (dims == 3) {
    out_shape = {in.at(0) + padding_after.at(0) + padding_before.at(0), 
                 in.at(1) + padding_after.at(1) + padding_before.at(1), 
                 in.at(2) + padding_after.at(2) + padding_before.at(2)};
  } else {
    UNIMPLEMENTED();
  }

  if (data_format == "channels_first") {
    out_shape.insert(out_shape.begin(), in_c);
  } else if (data_format == "channels_last") {
    out_shape.insert(out_shape.end(), in_c);
  } else {
    UNIMPLEMENTED();
  }
  out_shape.insert(out_shape.begin(), in_n);
  return Shape(out_shape);
}

void PadOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_shape = in_blob_desc->shape();
  int64_t num_axes = in_shape.NumAxes();
  const std::string data_format = op_conf().pad_conf().data_format();
  std::vector<int32_t> padding_before =
      GetPaddingsVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("padding_before"), num_axes - 2);
  std::vector<int32_t> padding_after =
      GetPaddingsVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("padding_after"), num_axes - 2);
  LOG(INFO) << "check point";
  if (data_format == "channels_first") {
    padding_before.insert(padding_before.begin(), 0);
    padding_after.insert(padding_after.begin(), 0);
  } else if (data_format == "channels_last") {
    padding_before.insert(padding_before.end(), 0);
    padding_after.insert(padding_after.end(), 0);
  } else {
    UNIMPLEMENTED();
  }
  padding_before.insert(padding_before.begin(), 0);
  padding_after.insert(padding_after.begin(), 0);
  
  FOR_RANGE(size_t, i, 0, num_axes) {
    kernel_conf->mutable_pad_conf()->mutable_padding_before()->Add(padding_before.at(i));
    kernel_conf->mutable_pad_conf()->mutable_padding_after()->Add(padding_after.at(i));
  }

}


std::vector<int32_t> PadOp::GetPaddingsVecInOpConf(const PbRf<int32_t>& field_vals, int32_t NDims) const {
  std::vector<int32_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      continue;
    } else {
      vec.push_back(field_vals.Get(index));
    }
  }
  return vec;
}

REGISTER_OP(OperatorConf::kPadConf, PadOp);

}  // namespace oneflow