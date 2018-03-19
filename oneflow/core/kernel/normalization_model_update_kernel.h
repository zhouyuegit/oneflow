#ifndef ONEFLOW_CORE_KERNEL_NORMALIZATION_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMALIZATION_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalizationModelUpdateKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationModelUpdateKernel);
  NormalizationModelUpdateKernel() = default;
  ~NormalizationModelUpdateKernel() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMALIZATION_MODEL_UPDATE_KERNEL_H_
