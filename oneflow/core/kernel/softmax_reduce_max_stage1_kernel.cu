#include "oneflow/core/kernel/softmax_reduce_max_stage1_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void SetMaskGpu(const int32_t n, const int32_t count, const T* in, const T* out, int32_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) { mask[i] = (in[i] == out[i / count]) ? 1 : 0; }
}

template<typename T>
__global__ void SetWithMaskGpu(const int32_t n, const int32_t count, const T* out_diff, const int32_t* mask, const int32_t* max_count, const int32_t* global_max_count,
                                T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {in_diff[i] = out_diff[i / count] * mask[i] * max_count[i] / global_max_count[i / count];
  }
  
}
}  // namespace

template<typename T>
struct SoftmaxReduceMaxStage1KernelUtil<DeviceType::kGPU, T> {
  static void SetMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* in, const T* out, int32_t* mask) {
      SetMaskGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, count, in, out, mask);
  }


  static void SetWithMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* out_diff, const int32_t* mask, const int32_t* max_count, const int32_t* global_max_count,
                                T* in_diff){
      SetWithMaskGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, count, out_diff, mask, max_count, global_max_count, in_diff);
    }
};

#define INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxReduceMaxStage1KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
