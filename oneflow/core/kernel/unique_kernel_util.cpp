#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<typename T, typename U>
struct UniqueKernelUtil<DeviceType::kCPU, T, U> {
  static void Unique(DeviceCtx* ctx, int64_t n, const T* in, int64_t* num_unique, T* unique_out,
                     U* idx_out, void* workspace, int64_t workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU(k_type_pair, v_type_pair)                \
  template struct UniqueKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(k_type_pair), \
                                   OF_PP_PAIR_FIRST(v_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU, UNIQUE_KERNEL_KV_DATA_TYPE_SEQ,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU

}  // namespace oneflow
