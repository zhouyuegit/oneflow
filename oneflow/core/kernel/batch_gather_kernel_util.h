#ifndef ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct BatchGatherKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const T* in, const K* indices, const Shape& flat_out_shape,
                      int64_t gather_dim_size, T* out);
};

template<DeviceType device_type, typename T>
struct BatchGatherKernelUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* in, const Blob* indices, Blob* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_UTIL_H_
