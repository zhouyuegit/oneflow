#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SoftmaxReduceMaxStage0Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxReduceMaxStage0Kernel);
  SoftmaxReduceMaxStage0Kernel() = default;
  ~SoftmaxReduceMaxStage0Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct SoftmaxReduceMaxStage0KernelUtil {
  static void SetMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* in, const T* max, int32_t* mask);
  static void SetWithMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* max_diff, const int32_t* mask, const int32_t* max_count,
                                T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_MEAN_KERNEL_H_
