#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SoftmaxReduceMaxStage1Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxReduceMaxStage1Kernel);
  SoftmaxReduceMaxStage1Kernel() = default;
  ~SoftmaxReduceMaxStage1Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct SoftmaxReduceMaxStage1KernelUtil {
  static void SetMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* in, const T* out, int32_t* mask);
  static void SetWithMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* out_diff, const int32_t* mask, const int32_t* max_count, const int32_t* global_max_count,
                                T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_MEAN_KERNEL_H_
