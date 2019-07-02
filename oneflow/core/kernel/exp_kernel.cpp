#include "oneflow/core/kernel/exp_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ExpKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Exp(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                   in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ExpKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->static_shape().elem_cnt(), out_diff_blob->dptr<T>(), out_blob->dptr<T>(),
                                      in_diff_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kExpConf, ExpKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
