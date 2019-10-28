#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

int64_t GetUniqueIdxSize(int64_t n) { return GetCudaAlignedSize(n * sizeof(int64_t)); }

template<DeviceType device_type, typename K, typename T>
void IndexedSlicesReduceSumKernelUtil<device_type, K, T>::ReduceSum(
    DeviceCtx* ctx, int64_t n, int64_t m, const K* indices, const T* values,
    int64_t* num_unique_indices, K* indices_out, T* values_out, void* workspace,
    int64_t workspace_size_in_bytes) {
  const int64_t unique_idx_size = GetUniqueIdxSize(n);
  CHECK_LE(unique_idx_size, workspace_size_in_bytes);
  int64_t* unique_idx_ptr = reinterpret_cast<int64_t*>(workspace);
  void* unique_workspace_ptr = reinterpret_cast<unsigned char*>(workspace) + unique_idx_size;
  const int64_t unique_workspace_size = workspace_size_in_bytes - unique_idx_size;
  UniqueKernelUtil<device_type, K, int64_t>::Unique(ctx, n, indices, num_unique_indices,
                                                    indices_out, unique_idx_ptr,
                                                    unique_workspace_ptr, unique_workspace_size);
  const Shape flat_in_shape({1, n, m});
  Memset<device_type>(ctx, values_out, 0, n * m * sizeof(T));
  GatherKernelUtilImpl<device_type, T, int64_t>::Backward(ctx, unique_idx_ptr, n, values,
                                                          flat_in_shape, values_out, 0);
}

template<DeviceType device_type, typename K, typename T>
void IndexedSlicesReduceSumKernelUtil<device_type, K, T>::GetWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t m, int64_t* workspace_size_in_bytes) {
  int64_t unique_workspace_size;
  UniqueKernelUtil<device_type, K, int64_t>::GetWorkspaceSizeInBytes(ctx, n,
                                                                     &unique_workspace_size);
  *workspace_size_in_bytes = GetUniqueIdxSize(n) + unique_workspace_size;
}

#define INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL(device_type, key_type_pair,            \
                                                          val_type_pair)                         \
  template struct IndexedSlicesReduceSumKernelUtil<device_type, OF_PP_PAIR_FIRST(key_type_pair), \
                                                   OF_PP_PAIR_FIRST(val_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL

}  // namespace oneflow
