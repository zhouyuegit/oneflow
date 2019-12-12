#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"

namespace oneflow {

template<>
struct BatchMemcpyKernelUtil<DeviceType::kCPU> {
  static void Copy(DeviceCtx* ctx, const BatchMemcpyParams& params) { UNIMPLEMENTED(); }
};

template struct BatchMemcpyKernelUtil<DeviceType::kCPU>;

}  // namespace oneflow
