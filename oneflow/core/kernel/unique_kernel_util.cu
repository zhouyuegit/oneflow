#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

namespace {

int64_t WorkspaceSizeAlign(int64_t size) { return RoundUp(size, kCudaAlignSize); }

}  // namespace

template<typename T, typename U>
struct UniqueKernelUtil<DeviceType::kGPU, T, U> {
  void Unique(DeviceCtx* ctx, int64_t n, const T* in, int64_t* num_unique, T* unique_out,
              U* idx_out, void* workspace, int64_t workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
  void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes);
};

template<typename T, typename U>
void UniqueKernelUtil<DeviceType::kGPU, T, U>::GetUniqueWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  *workspace_size_in_bytes = 0;
  {}
}

}  // namespace oneflow
