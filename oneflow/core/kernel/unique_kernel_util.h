#ifndef ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename U>
struct UniqueKernelUtil {
  void Unique(DeviceCtx* ctx, int64_t n, const T* in, int64_t* num_unique, T* unique_out,
              U* idx_out, void* workspace, int64_t workspace_size_in_bytes);
  void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_
