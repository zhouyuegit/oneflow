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
int64_t GetCubSortTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  CudaCheck(cub::DeviceRadixSort::SortPairs<T, U>(nullptr, cub_sort_temp_store_size, nullptr,
                                                  nullptr, nullptr, nullptr, n));
  CHECK_GE(cub_sort_temp_store_size, 0);
  CHECK_LT(cub_sort_temp_store_size, GetMaxVal<int64_t>());
  return SizeAlign(static_cast<int64_t>(cub_sort_temp_store_size));
}

template<typename T, typename U>
int64_t GetCubRleTempStorageSize(int64_t n) {
  size_t cub_rle_temp_store_size = 0;
  CudaCheck(cub::DeviceRunLengthEncode::Encode<T*, T*, U*, int64_t*>(
      nullptr, cub_rle_temp_store_size, nullptr, nullptr, nullptr, nullptr, n));
  CHECK_GE(cub_rle_temp_store_size, 0);
  CHECK_LT(cub_rle_temp_store_size, GetMaxVal<int64_t>());
  return SizeAlign(static_cast<int64_t>(cub_rle_temp_store_size));
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
  const int64_t sort_temp_storage_size = GetCubSortTempStorageSize<T, U>(n);
  const int64_t rle_tmp_storage_size = GetCubRleTempStorageSize<T, U>(n);

  *workspace_size_in_bytes =
      sort_key_out_size + sort_value_in_size + sort_value_out_size + sort_temp_storage_size;
}

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU(k_type_pair, v_type_pair)                \
  template struct UniqueKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(k_type_pair), \
                                   OF_PP_PAIR_FIRST(v_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU, UNIQUE_KERNEL_KV_DATA_TYPE_SEQ,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU

}  // namespace oneflow
