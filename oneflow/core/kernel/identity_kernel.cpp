#include "oneflow/core/kernel/identity_kernel.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int, i, 0, in_blob->shape().elem_cnt()) {
    const TensorBuffer* in_tb_i = in_blob->dptr<TensorBuffer>() + i;
    const TensorBuffer* out_tb_i = out_blob->dptr<TensorBuffer>() + i;
    LOG(INFO) << "Identity in_blob " << i << "th TensorBuffer " << in_tb_i;
    LOG(INFO) << "Identity out_blob " << i << "th TensorBuffer " << out_tb_i;
    LOG(INFO) << "Identity in_blob " << i << "th TensorBuffer MemoryCase " << &in_tb_i->mem_case();
    LOG(INFO) << "Identity out_blob " << i << "th TensorBuffer MemoryCase "
              << &out_tb_i->mem_case();
    LOG(INFO) << "Identity in_blob " << i << "th TensorBuffer MemoryCase host_mem "
              << &in_tb_i->mem_case().host_mem();
    LOG(INFO) << "Identity out_blob " << i << "th TensorBuffer MemoryCase host_mem "
              << &out_tb_i->mem_case().host_mem();
  }
  BnInOp2Blob("out")->CopyValidDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kIdentityConf, IdentityKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kCopyConf, IdentityKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kParallelCastConf, IdentityKernel);

}  // namespace oneflow
