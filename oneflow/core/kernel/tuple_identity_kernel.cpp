#include "oneflow/core/kernel/tuple_identity_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  const auto& output_bns = this->op_attribute().output_bns();
  CHECK_EQ(input_bns.size(), output_bns.size());
  FOR_RANGE(int, i, 0, input_bns.size()) {
    const Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    if (in_blob->has_dim0_valid_num_field()) {
      CHECK(out_blob->has_dim0_valid_num_field());
      CHECK_EQ(in_blob->dim0_inner_shape(), out_blob->dim0_inner_shape());
      out_blob->CopyDim0ValidNumFrom(ctx.device_ctx, in_blob);
    }
  }
}

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  const auto& output_bns = this->op_attribute().output_bns();
  CHECK_EQ(input_bns.size(), output_bns.size());
  FOR_RANGE(int, i, 0, input_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    out_blob->CopyValidDataContentFrom(ctx.device_ctx, BnInOp2Blob(input_bns.Get(i)));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTupleIdentityConf, TupleIdentityKernel);

}  // namespace oneflow
