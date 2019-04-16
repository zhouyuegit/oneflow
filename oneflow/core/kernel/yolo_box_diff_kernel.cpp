#include "oneflow/core/kernel/yolo_box_diff_kernel.h"

namespace oneflow {

template<typename T>
void YoloBoxDiffKernel<DeviceType::kCPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void YoloBoxDiffKernel<DeviceType::kCPU, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void YoloBoxDiffKernel<DeviceType::kCPU, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloBoxDiffConf, YoloBoxDiffKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow