#ifndef ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class YoloProbLossKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloProbLossKernel);
  YoloProbLossKernel() = default;
  ~YoloProbLossKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void CalcProbLoss(const int64_t im_index,
                    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void CalSub(const int32_t n, const int32_t* label_ptr, const T* pred_ptr, T* loss_ptr) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_
