#ifndef ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

template<typename T>
class MaskTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskTargetKernel);
  MaskTargetKernel() = default;
  ~MaskTargetKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void GetMaskBoxes(const std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  int32_t GetMaxOverlapMaskBoxIndex(
      T im_index, int32_t roi_index,
      const std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void Polys2MaskWrtBox(T im_index, int32_t gt_index, int32_t roi_index,
                        const std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_