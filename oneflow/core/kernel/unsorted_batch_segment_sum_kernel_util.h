#ifndef ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct UnsortedBatchSegmentSumKernelUtil final {
  static void Dispatch(DeviceCtx* ctx, int64_t num_batches, int64_t num_indices,
                       int64_t num_segments, int64_t instance_size, const K* indices, const T* in,
                       T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_H_
