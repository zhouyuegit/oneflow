#include "oneflow/core/kernel/softmax_loss_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t batch_idx, const K* label, const int64_t labels_num,
                             const int64_t lower_bound) {
  const int64_t idx = label[batch_idx] - lower_bound;
  if (idx >= 0 && idx < labels_num) {
    return batch_idx * labels_num + idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void SoftmaxLossGradBackwardGpu(const int64_t batch_num, const K* label,
                                           const int64_t labels_num, const int64_t lower_bound,
                                           T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, batch_num) {
    const int64_t idx = GetOffset<K>(i, label, labels_num, lower_bound);
    if (idx != -1) { in_diff[idx] = in_diff[idx] - 1; }
  }
}

}  // namespace

template<typename T, typename K>
struct SoftmaxLossGradKernelUtil<DeviceType::kGPU, T, K> final {
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const K* label, const int64_t lower_bound, T* in_diff);
};

template<typename T, typename K>
void SoftmaxLossGradKernelUtil<DeviceType::kGPU, T, K>::Backward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const K* label,
    const int64_t lower_bound, T* in_diff) {
  SoftmaxLossGradBackwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(batch_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          batch_num, label, labels_num, lower_bound, in_diff);
}

#define MAKE_SOFTMAX_LOSS_GRAD_KERNEL_UTIL_ENTRY(in_type_pair, index_type_pair)               \
  template struct SoftmaxLossGradKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                            OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_SOFTMAX_LOSS_GRAD_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef MAKE_SOFTMAX_LOSS_GRAD_KERNEL_UTIL_ENTRY

}  // namespace oneflow
