#ifndef ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UITL_H_
#define ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UITL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

constexpr int32_t kBatchMemcpyMaxParam = 16;
constexpr int32_t kBatchMemcpyMaxSize = 2 * 1024 * 1024;

struct BatchMemcpyParams {
  void* dst[kBatchMemcpyMaxParam];
  const void* src[kBatchMemcpyMaxParam];
  int32_t size[kBatchMemcpyMaxParam];
  int32_t num_params;
};

template<DeviceType device_type>
struct BatchMemcpyKernelUtil {
  static void Copy(DeviceCtx* ctx, const BatchMemcpyParams& params);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UITL_H_
