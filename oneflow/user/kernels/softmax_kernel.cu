/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/softmax_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int64_t kSoftmaxGpuBlockDim = 256;

template<typename T>
struct SoftmaxUtil {
  using ComputeType = T;
  __device__ static ComputeType ToComputeType(T v) { return v; }
  __device__ static T FromComputeType(ComputeType v) { return v; }
};

template<>
struct SoftmaxUtil<half> {
  using ComputeType = float;
  __device__ static ComputeType ToComputeType(half v) { return __half2float(v); }
  __device__ static half FromComputeType(ComputeType v) { return __float2half(v); }
};

template<typename T>
int GetForwardDynamicSharedMemorySize(const int w) {
  return w * sizeof(typename SoftmaxUtil<T>::ComputeType);
}

template<typename T>
int GetBackwardDynamicSharedMemorySize(const int w) {
  return 2 * w * sizeof(typename SoftmaxUtil<T>::ComputeType);
}

template<typename T>
__global__ void SoftmaxGpuForwardImpl(const int n, const int w, const T* in, T* prob) {
  using Util = SoftmaxUtil<T>;
  using ComputeType = typename Util::ComputeType;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char fw_shared_buf[];
  auto* compute_buf = reinterpret_cast<ComputeType*>(fw_shared_buf);
  __shared__ ComputeType row_reduce_result;
  typedef cub::BlockReduce<ComputeType, kSoftmaxGpuBlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    ComputeType thread_max = std::numeric_limits<ComputeType>::lowest();
    const int row_offset = row * w;
    const T* in_row = in + row_offset;
    T* prob_row = prob + row_offset;
    for (int col = tid; col < w; col += blockDim.x) {
      const ComputeType v = Util::ToComputeType(in_row[col]);
      compute_buf[col] = v;
      thread_max = max(thread_max, v);
    }
    __syncthreads();
    ComputeType block_max = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_max, cub::Max());
    if (tid == 0) { row_reduce_result = block_max; }
    __syncthreads();
    const ComputeType row_max_t = row_reduce_result;
    ComputeType thread_sum = 0;
    for (int col = tid; col < w; col += blockDim.x) {
      const ComputeType exp_v = expf(compute_buf[col] - row_max_t);
      compute_buf[col] = exp_v;
      thread_sum += exp_v;
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    if (tid == 0) { row_reduce_result = block_sum; }
    __syncthreads();
    const ComputeType row_sum_t = row_reduce_result;
    for (int col = tid; col < w; col += blockDim.x) {
      prob_row[col] = Util::FromComputeType(compute_buf[col] / row_sum_t);
    }
  }
}

template<typename T>
void SoftmaxForwardGpu(DeviceCtx* ctx, const int n, const int w, const T* in, T* prob) {
  const int block_num = std::max(static_cast<int>(n), kCudaMaxBlocksNum);
  SoftmaxGpuForwardImpl<<<block_num, kSoftmaxGpuBlockDim, GetForwardDynamicSharedMemorySize<T>(w),
                          ctx->cuda_stream()>>>(n, w, in, prob);
}

template<>
void SoftmaxForwardGpu<float16>(DeviceCtx* ctx, const int n, const int w, const float16* in,
                                float16* prob) {
  SoftmaxForwardGpu<half>(ctx, n, w, reinterpret_cast<const half*>(in),
                          reinterpret_cast<half*>(prob));
}

template<typename T>
__global__ void SoftmaxGpuBackwardImpl(const int n, const int w, const T* dy, const T* prob,
                                       T* dx) {
  using Util = SoftmaxUtil<T>;
  using ComputeType = typename Util::ComputeType;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char bw_shared_buf[];
  auto* dy_buf = reinterpret_cast<ComputeType*>(bw_shared_buf);
  auto* prob_buf = reinterpret_cast<ComputeType*>(bw_shared_buf + w);
  __shared__ ComputeType row_reduce_result;
  typedef cub::BlockReduce<ComputeType, kSoftmaxGpuBlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    const int row_offset = row * w;
    const T* dy_row = dy + row_offset;
    const T* prob_row = prob + row_offset;
    T* dx_row = dx + row_offset;
    ComputeType thread_sum = 0;
    for (int col = tid; col < w; col += blockDim.x) {
      const ComputeType dy_v = Util::ToComputeType(dy_row[col]);
      const ComputeType prob_v = Util::ToComputeType(prob_row[col]);
      dy_buf[col] = dy_v;
      prob_buf[col] = prob_v;
      thread_sum += dy_v * prob_v;
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    if (tid == 0) { row_reduce_result = block_sum; }
    __syncthreads();
    const ComputeType row_sum_t = row_reduce_result;
    for (int col = tid; col < w; col += blockDim.x) {
      dx_row[col] = Util::FromComputeType((dy_buf[col] - row_sum_t) * prob_buf[col]);
    }
  }
}

template<typename T>
void SoftmaxBackwardGpu(DeviceCtx* ctx, const int n, const int w, const T* in, const T* prob,
                        T* dx) {
  const int block_num = std::max(static_cast<int>(n), kCudaMaxBlocksNum);
  SoftmaxGpuBackwardImpl<<<block_num, kSoftmaxGpuBlockDim, GetBackwardDynamicSharedMemorySize<T>(w),
                           ctx->cuda_stream()>>>(n, w, in, prob, dx);
}

template<>
void SoftmaxBackwardGpu<float16>(DeviceCtx* ctx, const int n, const int w, const float16* in,
                                 const float16* prob, float16* dx) {
  SoftmaxBackwardGpu<half>(ctx, n, w, reinterpret_cast<const half*>(in),
                           reinterpret_cast<const half*>(prob), reinterpret_cast<half*>(dx));
}

template<typename T>
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t num_classes = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t num_instances = in_shape.Count(0, in_shape.NumAxes() - 1);
    SoftmaxForwardGpu<T>(ctx->device_ctx(), num_instances, num_classes, in->dptr<T>(),
                         out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GPU_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("softmax")                                                                \
      .SetCreateFn<SoftmaxKernel<dtype>>()                                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                             \
        const int64_t num_classes = in_shape->At(in_shape->NumAxes() - 1);                       \
        const int64_t num_instances = in_shape->Count(0, in_shape->NumAxes() - 1);               \
        return SoftmaxKernelUtil<DeviceType::kGPU, dtype>::GetComputeProbTempStorageSizeInBytes( \
            num_instances, num_classes);                                                         \
      });

REGISTER_SOFTMAX_GPU_KERNEL(float16)
REGISTER_SOFTMAX_GPU_KERNEL(float)
REGISTER_SOFTMAX_GPU_KERNEL(double)
#undef REGISTER_SOFTMAX_GPU_KERNEL

template<typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t num_instances = y->shape().elem_cnt() / num_classes;

    SoftmaxBackwardGpu<T>(ctx->device_ctx(), num_instances, num_classes, dy->dptr<T>(),
                          y->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GRAD_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("softmax_grad")                                                           \
      .SetCreateFn<SoftmaxGradKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))           \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);                             \
        const int64_t num_classes = dy_shape->At(dy_shape->NumAxes() - 1);                       \
        const int64_t num_instances = dy_shape->Count(0, dy_shape->NumAxes() - 1);               \
        return SoftmaxKernelUtil<DeviceType::kGPU, dtype>::GetComputeProbTempStorageSizeInBytes( \
            num_instances, num_classes);                                                         \
      });

REGISTER_SOFTMAX_GRAD_KERNEL(float16)
REGISTER_SOFTMAX_GRAD_KERNEL(float)
REGISTER_SOFTMAX_GRAD_KERNEL(double)
#undef REGISTER_SOFTMAX_GRAD_KERNEL

}  // namespace

}  // namespace oneflow
