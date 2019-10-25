#include "oneflow/core/kernel/indexed_slices_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename K, typename T>
void IndexedSlicesKernelUtil<device_type, K, T>::ReduceSumByKey(
    DeviceCtx* ctx, int64_t n, int64_t m, const K* indices, const T* values,
    int64_t* num_unique_indices, const K* indices_out, T* values_out, void* workspace,
    int64_t workspace_size_in_bytes) {}

template<DeviceType device_type, typename K, typename T>
void IndexedSlicesKernelUtil<device_type, K, T>::GetReduceSumByKeyWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t m, int64_t* workspace_size_in_bytes) {}

#define INSTANTIATE_INDEXED_SLICES_KERNEL_UTIL(device_type, key_type_pair, val_type_pair) \
  template struct IndexedSlicesKernelUtil<device_type, OF_PP_PAIR_FIRST(key_type_pair),   \
                                          OF_PP_PAIR_FIRST(val_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_KERNEL_UTIL

}  // namespace oneflow
