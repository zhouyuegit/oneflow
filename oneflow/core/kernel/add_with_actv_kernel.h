#ifndef ONEFLOW_CORE_KERNEL_ADD_WITH_ACTV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADD_WITH_ACTV_KERNEL_H_

#include "oneflow/core/kernel/add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AddWithActvKernel final : public KernelIfWithActivation<device_type, T>,
                                public AddKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddWithActvKernel);
  AddWithActvKernel() = default;
  ~AddWithActvKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADD_WITH_ACTV_KERNEL_H_
