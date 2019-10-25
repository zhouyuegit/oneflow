#include "oneflow/core/kernel/unique_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

int64_t SizeAlign(int64_t size) { return RoundUp(size, kCudaAlignSize); }

template<typename T, typename U>
int64_t GetSortKeySize(int64_t n) {
  return SizeAlign(n * sizeof(T));
}

template<typename T, typename U>
int64_t GetSortValueSize(int64_t n) {
  return SizeAlign(n * sizeof(U));
}

template<typename T, typename U>
int64_t GetSortTempStorageSize(int64_t n) {
  int64_t cub_sort_temp_store_size = 0;
  CudaCheck(cub::DeviceRadixSort::SortPairs<T, U>(nullptr, cub_sort_temp_store_size, nullptr,
                                                  nullptr, nullptr, nullptr, n));
  return cub_sort_temp_store_size;
}

}  // namespace

template<typename T, typename U>
struct UniqueKernelUtil<DeviceType::kGPU, T, U> {
  static void Unique(DeviceCtx* ctx, int64_t n, const T* in, int64_t* num_unique, T* unique_out,
                     U* idx_out, void* workspace, int64_t workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes);
};

template<typename T, typename U>
void UniqueKernelUtil<DeviceType::kGPU, T, U>::GetUniqueWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  const int64_t sort_key_out_size = GetSortKeySize<T, U>(n);
  const int64_t sort_value_in_size = GetSortValueSize<T, U>(n);
  const int64_t sort_value_out_size = GetSortValueSize<T, U>(n);
  const int64_t sort_temp_storage_size = GetSortTempStorageSize<T, U>(n);

  *workspace_size_in_bytes =
      sort_key_out_size + sort_value_in_size + sort_value_out_size + sort_temp_storage_size;
}

}  // namespace oneflow
