#ifndef ONEFLOW_CORE_KERNEL_YOLO_BOX_DIFF_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_YOLO_BOX_DIFF_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class YoloBoxDiffKernel;

template<typename T>
class YoloBoxDiffKernel<DeviceType::kCPU, T> final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloBoxDiffKernel);
  YoloBoxDiffKernel() = default;
  ~YoloBoxDiffKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<typename T>
class YoloBoxDiffKernel<DeviceType::kGPU, T> final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloBoxDiffKernel);
  YoloBoxDiffKernel() = default;
  ~YoloBoxDiffKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_YOLO_DIFF_LOSS_KERNEL_H_
