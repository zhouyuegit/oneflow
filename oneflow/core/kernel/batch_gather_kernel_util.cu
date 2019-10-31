#include "oneflow/core/kernel/batch_gather_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetInOffset(const int64_t out_offset, const K* indices,
                               const int64_t indices_num, const int64_t instance_size,
                               const int64_t gather_dim_size) {
  const int64_t batch_idx = out_offset / (indices_num * instance_size);
  const int64_t indices_idx = out_offset % (indices_num * instance_size) / instance_size;
  const int64_t inner_idx = out_offset % instance_size;
  const int64_t idx = indices[batch_idx * indices_num + indices_idx];
  assert(idx >= 0 && idx < gather_dim_size);
  return batch_idx * gather_dim_size * instance_size + idx * instance_size + inner_idx;
}

template<typename T, typename K>
__global__ void BatchGatherForwardGpu(const int64_t elem_cnt, const T* in, const K* indices,
                                      const int64_t indices_num, const int64_t instance_size,
                                      const int64_t gather_dim_size, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out[i] = in[GetInOffset<K>(i, indices, indices_num, instance_size, gather_dim_size)];
  }
}

template<typename T>
struct SharedMemory {
  // Ensure that we won't compile any un-specialized types
  __device__ T* getPointer() {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

template<>
struct SharedMemory<int> {
  __device__ int* getPointer() {
    extern __shared__ int s_int[];
    return s_int;
  }
};

template<>
struct SharedMemory<unsigned int> {
  __device__ unsigned int* getPointer() {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

template<>
struct SharedMemory<char> {
  __device__ char* getPointer() {
    extern __shared__ char s_char[];
    return s_char;
  }
};

template<>
struct SharedMemory<unsigned char> {
  __device__ unsigned char* getPointer() {
    extern __shared__ unsigned char s_uchar[];
    return s_uchar;
  }
};

template<>
struct SharedMemory<short> {
  __device__ short* getPointer() {
    extern __shared__ short s_short[];
    return s_short;
  }
};

template<>
struct SharedMemory<unsigned short> {
  __device__ unsigned short* getPointer() {
    extern __shared__ unsigned short s_ushort[];
    return s_ushort;
  }
};

template<>
struct SharedMemory<long> {
  __device__ long* getPointer() {
    extern __shared__ long s_long[];
    return s_long;
  }
};

template<>
struct SharedMemory<unsigned long> {
  __device__ unsigned long* getPointer() {
    extern __shared__ unsigned long s_ulong[];
    return s_ulong;
  }
};

template<>
struct SharedMemory<bool> {
  __device__ bool* getPointer() {
    extern __shared__ bool s_bool[];
    return s_bool;
  }
};

template<>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template<>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template<typename T, typename K>
__global__ void BatchGatherBackwardGpuV2(const int64_t batch_num, const int64_t indices_num,
                                         const int64_t gather_dim_size, const int64_t instance_size,
                                         const K* indices, const T* out_diff, T* in_diff) {
  SharedMemory<T> shared;
  T* buf = shared.getPointer();
  const int64_t in_diff_batch_instance_size = gather_dim_size * instance_size;
  const int64_t out_diff_batch_instance_size = indices_num * instance_size;
  for (int32_t batch_idx = blockIdx.x; batch_idx < batch_num; batch_idx += gridDim.x) {
    const K* batch_indices = indices + batch_idx * indices_num;
    const T* batch_out_diff = out_diff + batch_idx * out_diff_batch_instance_size;
    T* batch_in_diff = in_diff + batch_idx * in_diff_batch_instance_size;
    for (int32_t i = threadIdx.x; i < in_diff_batch_instance_size; i += blockDim.x) { buf[i] = 0; }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < out_diff_batch_instance_size; i += blockDim.x) {
      T val = batch_out_diff[i];
      if (val != 0) {
        gpu_atomic_add(buf + batch_indices[i / instance_size] * instance_size + i % instance_size,
                       val);
      }
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < in_diff_batch_instance_size; i += blockDim.x) {
      batch_in_diff[i] = buf[i];
    }
  }
}

template<typename T, typename K>
__global__ void BatchGatherBackwardGpu(const int64_t elem_cnt, const T* out_diff, const K* indices,
                                       const int64_t indices_num, const int64_t instance_size,
                                       const int64_t gather_dim_size, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T diff_val = out_diff[i];
    if (diff_val != static_cast<T>(0)) {
      gpu_atomic_add(
          in_diff + GetInOffset<K>(i, indices, indices_num, instance_size, gather_dim_size),
          diff_val);
    }
  }
}

}  // namespace

template<typename T, typename K>
struct BatchGatherKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const T* in, const K* indices, const Shape& flat_out_shape,
                      const int64_t gather_dim_size, T* out);
  static void Backward(DeviceCtx* ctx, const T* out_diff, const K* indices,
                       const Shape& flat_out_diff_shape, const int64_t gather_dim_size, T* in_diff);
};

template<typename T, typename K>
void BatchGatherKernelUtilImpl<DeviceType::kGPU, T, K>::Forward(DeviceCtx* ctx, const T* in,
                                                                const K* indices,
                                                                const Shape& flat_out_shape,
                                                                const int64_t gather_dim_size,
                                                                T* out) {
  const int64_t batch_num = flat_out_shape.At(0);
  const int64_t indices_num = flat_out_shape.At(1);
  const int64_t instance_size = flat_out_shape.At(2);

  const int64_t elem_cnt = batch_num * indices_num * instance_size;
  BatchGatherForwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, in, indices, indices_num, instance_size, gather_dim_size, out);
}

template<typename T, typename K>
void BatchGatherKernelUtilImpl<DeviceType::kGPU, T, K>::Backward(DeviceCtx* ctx, const T* out_diff,
                                                                 const K* indices,
                                                                 const Shape& flat_out_diff_shape,
                                                                 const int64_t gather_dim_size,
                                                                 T* in_diff) {
  const int64_t batch_num = flat_out_diff_shape.At(0);
  const int64_t indices_num = flat_out_diff_shape.At(1);
  const int64_t instance_size = flat_out_diff_shape.At(2);
  const int64_t elem_cnt = batch_num * indices_num * instance_size;
  const size_t out_batch_size_bytes = instance_size * gather_dim_size * sizeof(T);
  if (batch_num >= 256 && out_batch_size_bytes <= 16 * 1024 && indices_num * instance_size >= 256) {
    int32_t thread_num =
        std::min(static_cast<int32_t>(instance_size * indices_num), kCudaThreadsNumPerBlock);
    BatchGatherBackwardGpuV2<T, K><<<256, thread_num, out_batch_size_bytes, ctx->cuda_stream()>>>(
        batch_num, indices_num, gather_dim_size, instance_size, indices, out_diff, in_diff);
  } else {
    BatchGatherBackwardGpu<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, out_diff, indices, indices_num, instance_size, gather_dim_size, in_diff);
  }
}

#define INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_GPU(in_type_pair, index_type_pair)          \
  template struct BatchGatherKernelUtilImpl<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                            OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_GPU

}  // namespace oneflow
