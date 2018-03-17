#include "oneflow/core/micro_kernel/add_micro_kernel.h"

namespace oneflow {

template<typename T>
struct MicroKernelUtil<AddMicroKernel, DeviceType::kCPU, T> final {
  static void Forward(
      const AddMicroKernel<DeviceType::kCPU, T>* micro_kernel,
      const KernelCtx& device_ctx,
      const std::function<Blob*(const std::string&)>& Blob4BnInOp) {
    TODO();
  }

  static void Backward(
      const AddMicroKernel<DeviceType::kCPU, T>* micro_kernel,
      const KernelCtx& device_ctx,
      const std::function<Blob*(const std::string&)>& Blob4BnInOp) {
    TODO();
  }
};

#define INITIATE_MICRO_KERNEL(T, type_proto) \
  template struct MicroKernelUtil<AddMicroKernel, DeviceType::kCPU, T>;

OF_PP_FOR_EACH_TUPLE(INITIATE_MICRO_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
