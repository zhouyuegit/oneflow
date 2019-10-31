#include "oneflow/core/kernel/unsorted_batch_segment_sum_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename K>
void UnsortedBatchSegmentSumImplCpu(int64_t num_batches, int64_t num_indices, int64_t num_segments,
                                    int64_t instance_size, const K* indices, const T* in, T* out) {
  FOR_RANGE(int64_t, batch_idx, 0, num_batches) {
    const K* batch_indices = indices + batch_idx * num_indices;
    const T* batch_in = in + batch_idx * num_indices * instance_size;
    T* batch_out = out + batch_idx * num_segments * instance_size;
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = batch_indices[i];
      CHECK(idx >= 0 && idx < num_segments);
      const T* from = batch_in + i * instance_size;
      T* to = batch_out + idx * instance_size;
      std::transform(from, from + instance_size, to, to, std::plus<T>());
    }
  }
}

}  // namespace

template<typename T, typename K>
struct UnsortedBatchSegmentSumKernelUtil<DeviceType::kCPU, T, K> final {
  static void Dispatch(DeviceCtx* ctx, int64_t num_batches, int64_t num_indices,
                       int64_t num_segments, int64_t instance_size, const K* indices, const T* in,
                       T* out);
};

template<typename T, typename K>
void UnsortedBatchSegmentSumKernelUtil<DeviceType::kCPU, T, K>::Dispatch(
    DeviceCtx* ctx, int64_t num_batches, int64_t num_indices, int64_t num_segments,
    int64_t instance_size, const K* indices, const T* in, T* out) {
  Memset<DeviceType::kCPU>(ctx, out, 0, num_batches * num_segments * instance_size * sizeof(T));
  UnsortedBatchSegmentSumImplCpu(num_batches, num_indices, num_segments, instance_size, indices, in,
                                 out);
}

#define INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_CPU(data_type_pair, index_type_pair) \
  template struct UnsortedBatchSegmentSumKernelUtil<                                            \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_CPU

}  // namespace oneflow
