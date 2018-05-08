#ifndef ONEFLOW_CORE_KERNEL_MEAN_SQUARED_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MEAN_SQUARED_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class MeanSquaredLossKernel final : public LossKernel<device_type, PredType, LabelType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MeanSquaredLossKernel);
  MeanSquaredLossKernel() = default;
  ~MeanSquaredLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct MeanSquaredLossKernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t inst_num, const int64_t label_dim,
                      const LabelType* label, const PredType* pred, PredType* diff, PredType* loss);
  static void Backward(DeviceCtx* ctx, const int64_t inst_num, const int64_t label_dim,
                       const PredType* diff, PredType* pred_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MEAN_SQUARED_LOSS_KERNEL_H_
