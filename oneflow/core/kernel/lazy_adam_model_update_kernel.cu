#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lazy_adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, T l1, T l2, T beta1, T beta2, T epsilon, T* model_diff,
                               T* model, T* m, T* v, const float* local_learning_rate) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    if (abs(model_diff[i]) < 1e-12) { continue; }
    T reg_diff = RegDiff(model_diff[i], l1, l2, model[i]);
    m[i] = beta1 * m[i] + (1 - beta1) * reg_diff;
    v[i] = beta2 * v[i] + (1 - beta2) * reg_diff * reg_diff;
    model[i] = model[i] - (*local_learning_rate) * m[i] / (sqrt(v[i]) + epsilon);
  }
}

template<typename T>
__global__ void ComputeLocalLearningRateGpu(T beta1, T beta2, const int64_t* train_step,
                                            const float* learning_rate,
                                            float* local_learning_rate) {
  const T beta1_t = pow(beta1, *train_step + 1);
  const T beta2_t = pow(beta2, *train_step + 1);
  *local_learning_rate = *learning_rate * sqrt(1 - (beta2_t)) / (1 - (beta1_t));
}

}  // namespace

template<typename T>
class LazyAdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T l1, T l2,
                          T beta1, T beta2, T epsilon, const int64_t* train_step, T* beta1_t,
                          T* beta2_t, T* model_diff, T* model, T* m, T* v,
                          float* local_learning_rate) {
    if (train_step != nullptr) {
      ComputeLocalLearningRateGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(
          beta1, beta2, train_step, learning_rate, local_learning_rate);
      UpdateModelGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, l1, l2, beta1, beta2, epsilon, model_diff, model, m, v, local_learning_rate);
    } else {
      UpdateModelGpu<T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, l1, l2, beta1, beta2, epsilon, model_diff, model, m, v, learning_rate);
    }
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LazyAdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
