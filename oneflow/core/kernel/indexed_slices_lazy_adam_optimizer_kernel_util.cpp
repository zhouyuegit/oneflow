#include "oneflow/core/kernel/indexed_slices_lazy_adam_optimizer_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
struct IndexedSlicesLazyAdamOptimizerKernelUtil<DeviceType::kCPU, T, K> {
  static void UpdateModel(DeviceCtx* ctx, T l1, T l2, T beta1, T beta2, T epsilon,
                          int64_t num_instance, int64_t feature_size, int64_t lower_bound,
                          int64_t upper_bound, const int64_t* num_unique_instance,
                          const int64_t* train_step, const float* learning_rate, const K* indices,
                          const T* values, T* model, T* m, T* v) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_CPU(key_type_pair, \
                                                                       idx_type_pair) \
  template struct IndexedSlicesLazyAdamOptimizerKernelUtil<                           \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, UNIQUE_KERNEL_KV_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_CPU

}  // namespace oneflow
