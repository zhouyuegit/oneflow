#ifndef ONEFLOW_CORE_KERNEL_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const int64_t* train_step, const float* learning_rate, const K* indices,
                     const T* values, T* model, T* momentum);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_H_
