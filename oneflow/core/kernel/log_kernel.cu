#include "oneflow/core/kernel/log_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void LogGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = SafeLog(x[i]); }
}

}  // namespace

template<typename T>
struct LogKernelUtil<DeviceType::kGPU, T> {
  static void Log(DeviceCtx* ctx, const int64_t n, const T* x, T* y){
    LogGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, x, y);
  }
};

#define INSTANTIATE_LOG_KERNEL_UTIL(type_cpp, type_proto) \
  template struct LogKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LOG_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
