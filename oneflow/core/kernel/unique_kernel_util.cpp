#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<typename T, typename U>
struct UniqueKernelUtil<DeviceType::kGPU, T, U> {
  static void Unique(DeviceCtx* ctx, int64_t n, const T* in, int64_t* num_unique, T* unique_out,
                     U* idx_out, void* workspace, int64_t workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow
