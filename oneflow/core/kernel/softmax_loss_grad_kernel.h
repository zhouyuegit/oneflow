#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SoftmaxLossGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossGradKernel);
  SoftmaxLossGradKernel() = default;
  ~SoftmaxLossGradKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  int32_t lower_bound_;
};

template<DeviceType device_type, typename T, typename K>
struct SoftmaxLossGradKernelUtil final {
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const K* label, const int64_t lower_bound, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_GRAD_KERNEL_H_
