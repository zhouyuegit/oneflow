#ifndef ONEFLOW_CORE_KERNEL_YOLO_BOX_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_YOLO_BOX_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class YoloBoxKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloBoxKernel);
  YoloBoxKernel() = default;
  ~YoloBoxKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void WriteOutBBox(const T* bbox_ptr, IndexSequence& index_slice, T* out_bbox_ptr) const;
  void FilterAndSetProbs(const T* probs_ptr, IndexSequence& index_slice, T* out_probs_ptr) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_YOLO_BOX_KERNEL_H_
