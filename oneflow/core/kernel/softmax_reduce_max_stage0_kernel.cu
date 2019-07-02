#include "oneflow/core/kernel/softmax_reduce_max_stage0_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void SetMaskGpu(const int64_t n, const int64_t count, const T* in, const T* max, int32_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) { mask[i] = (in[i] == max[i / count]) ? 1 : 0; }
}

template<typename T>
__global__ void SetWithMaskGpu(const int64_t n, const int64_t count, const T* max_diff, const int32_t* mask, const int32_t* max_count,
                                T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {in_diff[i] = max_diff[i / count] * mask[i] / max_count[i / count];
  }
}

}  // namespace

template<typename T>
struct SoftmaxReduceMaxStage0KernelUtil<DeviceType::kGPU, T> {
  static void SetMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* in, const T* max, int32_t* mask) {
      SetMaskGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, count, in, max, mask);
  }


  static void SetWithMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* max_diff, const int32_t* mask, const int32_t* max_count,
                                T* in_diff){
      SetWithMaskGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, count, max_diff, mask, max_count, in_diff);
    }
};

#define INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxReduceMaxStage0KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
