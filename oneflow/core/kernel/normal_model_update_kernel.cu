#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CmptClipRatioByGlobalNormGpu(const T* global_norm_ptr, T clip_norm, T* ratio_ptr) {
  *ratio_ptr = clip_norm / max(*global_norm_ptr, clip_norm);
}

template<typename T>
__global__ void RegularizationGpu(const int64_t n, const T l1, const T l2,
    const T batch_instance_num, const T* model, T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model_diff[i] = RegDiff(model_diff[i], batch_instance_num, l1, l2, model[i]);
  }
}

}  // namespace

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void CmptClipRatioByGlobalNorm(DeviceCtx* ctx, const T* global_norm_ptr, T clip_norm,
                                        T* ratio_ptr) {
    CmptClipRatioByGlobalNormGpu<T>
        <<<1, 1, 0, ctx->cuda_stream()>>>(global_norm_ptr, clip_norm, ratio_ptr);
  }
  static void Regularization(const int64_t n, const T l1, const T l2, const T batch_instance_num, const T* model,
      T* model_diff) {
    RegularizationGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, l1, l2, batch_instance_num, model, model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NormalMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
