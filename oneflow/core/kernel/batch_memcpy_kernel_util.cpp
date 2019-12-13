#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"

namespace oneflow {

template<>
void BatchMemcpyKernelUtil<DeviceType::kCPU>::Copy(DeviceCtx* ctx,
                                                   const BatchMemcpyParams& params) {
  UNIMPLEMENTED();
}

}  // namespace oneflow
