#include "oneflow/core/kernel/logistic_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <math.h>
namespace oneflow {

namespace {

template<typename T>
__global__ void LogisticForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.f / (1.f + expf(-x[i])); }
}

}  // namespace

template<typename T>
struct LogisticKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    LogisticForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
  }
};

#define INSTANTIATE_Logistic_KERNEL_UTIL(type_cpp, type_proto) \
  template class LogisticKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_Logistic_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow