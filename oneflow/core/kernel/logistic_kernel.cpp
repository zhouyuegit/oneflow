#include "oneflow/core/kernel/logistic_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LogisticKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  KernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                      in_blob->dptr<T>(), BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void LogisticKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLogisticConf, LogisticKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
