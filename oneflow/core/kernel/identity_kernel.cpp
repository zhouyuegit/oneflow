#include "oneflow/core/kernel/identity_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type>
void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  const size_t copy_size = src->ByteSizeOfValidDataContent();
  CHECK_EQ(dst->ByteSizeOfValidDataContent(), copy_size);
  Memcpy<device_type>(ctx, dst->mut_dptr(), src->dptr(), copy_size);
}

}  // namespace

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob<device_type>(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kIdentityConf, IdentityKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kParallelCastConf, IdentityKernel);

}  // namespace oneflow
