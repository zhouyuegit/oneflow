#include "oneflow/core/kernel/indexed_slices_lazy_adam_optimizer_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void UpdateModelGpu(T l1, T l2, T beta1, T beta2, T epsilon, int64_t feature_size,
                               const int64_t* num_unique_instance, const int64_t* train_step,
                               const float* learning_rate, const K* indices, const T* values,
                               T* model, T* m, T* v) {
  const T beta1_t = pow(beta1, *train_step);
  const T beta2_t = pow(beta2, *train_step);
  const float local_learning_rate = *learning_rate * sqrt(1 - (beta2_t)) / (1 - (beta1_t));
  const int64_t n = *num_unique_instance * feature_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const K instance_id = indices[i / feature_size];
    const T diff = values[i];
    const K model_idx = instance_id * feature_size + i % feature_size;
    const T old_model = model[model_idx];
    T reg_diff = RegDiff(diff, l1, l2, old_model);
    m[model_idx] = beta1 * m[model_idx] + (1 - beta1) * reg_diff;
    v[model_idx] = beta2 * v[model_idx] + (1 - beta2) * reg_diff * reg_diff;
    model[model_idx] =
        old_model - local_learning_rate * m[model_idx] / (sqrt(v[model_idx]) + epsilon);
  }
}

}  // namespace

template<typename T, typename K>
struct IndexedSlicesLazyAdamOptimizerKernelUtil<DeviceType::kGPU, T, K> {
  static void UpdateModel(DeviceCtx* ctx, T l1, T l2, T beta1, T beta2, T epsilon,
                          int64_t num_instance, int64_t feature_size,
                          const int64_t* num_unique_instance, const int64_t* train_step,
                          const float* learning_rate, const K* indices, const T* values, T* model,
                          T* m, T* v);
};

template<typename T, typename K>
void IndexedSlicesLazyAdamOptimizerKernelUtil<DeviceType::kGPU, T, K>::UpdateModel(
    DeviceCtx* ctx, T l1, T l2, T beta1, T beta2, T epsilon, int64_t num_instance,
    int64_t feature_size, const int64_t* num_unique_instance, const int64_t* train_step,
    const float* learning_rate, const K* indices, const T* values, T* model, T* m, T* v) {
  UpdateModelGpu<T, K><<<BlocksNum4ThreadsNum(num_instance * feature_size), kCudaThreadsNumPerBlock,
                         0, ctx->cuda_stream()>>>(l1, l2, beta1, beta2, epsilon, feature_size,
                                                  num_unique_instance, train_step, learning_rate,
                                                  indices, values, model, m, v);
}

#define INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_GPU(key_type_pair, \
                                                                       idx_type_pair) \
  template struct IndexedSlicesLazyAdamOptimizerKernelUtil<                           \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_GPU,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ, UNIQUE_KERNEL_KV_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_OPTIMIZER_KERNEL_UTIL_GPU

}  // namespace oneflow
