#include "oneflow/core/kernel/unsorted_batch_segment_sum_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOutOffset(const int64_t in_offset, const K* indices,
                                const int64_t num_indices, const int64_t instance_size,
                                const int64_t num_segments) {
  const int64_t batch_idx = in_offset / (num_indices * instance_size);
  const int64_t indices_idx = in_offset % (num_indices * instance_size) / instance_size;
  const int64_t inner_idx = in_offset % instance_size;
  const int64_t idx = indices[batch_idx * num_indices + indices_idx];
  assert(idx >= 0 && idx < num_segments);
  return batch_idx * num_segments * instance_size + idx * instance_size + inner_idx;
}

template<typename T, typename K>
__global__ void ImplNaive(const int64_t elem_cnt, const int64_t num_indices,
                          const int64_t num_segments, const int64_t instance_size, const K* indices,
                          const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T val = in[i];
    if (val != 0) {
      gpu_atomic_add(out + GetOutOffset<K>(i, indices, num_indices, instance_size, num_segments),
                     val);
    }
  }
}

constexpr int64_t kImplBatchWiseMinNumBatches = 128;
constexpr int64_t kImplBatchWiseMaxSharedBufSizeInBytes = 16 * 1024;
constexpr int64_t kImplBatchWiseMinInBatchSize = 256;
constexpr int32_t kImplBatchWiseMaxThreadNum = 512;
constexpr int32_t kImplBatchWiseMaxBlockNum = 256;

template<typename T>
struct GpuImplBatchWiseSharedBufferHelper {
  __device__ T* GetPtr();
};

template<>
struct GpuImplBatchWiseSharedBufferHelper<float> {
  __device__ float* GetPtr() {
    extern __shared__ float batch_wise_shared_buffer_f[];
    return batch_wise_shared_buffer_f;
  }
};

template<>
struct GpuImplBatchWiseSharedBufferHelper<double> {
  __device__ double* GetPtr() {
    extern __shared__ double batch_wise_shared_buffer_d[];
    return batch_wise_shared_buffer_d;
  }
};

template<typename T, typename K>
__global__ void ImplBatchWise(const int64_t num_batches, const int64_t num_indices,
                              const int64_t num_segments, const int64_t instance_size,
                              const K* indices, const T* in, T* out) {
  T* buf = GpuImplBatchWiseSharedBufferHelper<T>().GetPtr();
  const int64_t in_batch_size = num_indices * instance_size;
  const int64_t out_batch_size = num_segments * instance_size;
  for (int64_t batch_idx = blockIdx.x; batch_idx < num_batches; batch_idx += gridDim.x) {
    const K* batch_indices = indices + batch_idx * num_indices;
    const T* batch_in = in + batch_idx * in_batch_size;
    T* batch_out = out + batch_idx * out_batch_size;
    for (int64_t i = threadIdx.x; i < out_batch_size; i += blockDim.x) { buf[i] = 0; }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < in_batch_size; i += blockDim.x) {
      T val = batch_in[i];
      if (val != 0) {
        gpu_atomic_add(buf + batch_indices[i / instance_size] * instance_size + i % instance_size,
                       val);
      }
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < out_batch_size; i += blockDim.x) { batch_out[i] = buf[i]; }
  }
}

int32_t ImplBatchWiseGetThreadNum(const int64_t in_batch_size) {
  const int32_t thread_num = std::min(in_batch_size, static_cast<int64_t>(GetMaxVal<int32_t>()));
  const int32_t max_thread_num = std::min(kImplBatchWiseMaxThreadNum, kCudaThreadsNumPerBlock);
  return std::min(thread_num, max_thread_num);
}

int32_t ImplBatchWiseGetBlockNum(const int64_t num_batches) {
  const int32_t block_num = std::min(num_batches, static_cast<int64_t>(GetMaxVal<int32_t>()));
  const int32_t max_block_num = std::min(kImplBatchWiseMaxBlockNum, kCudaMaxBlocksNum);
  return std::min(block_num, max_block_num);
}

}  // namespace

template<typename T, typename K>
struct UnsortedBatchSegmentSumKernelUtil<DeviceType::kGPU, T, K> final {
  static void Dispatch(DeviceCtx* ctx, int64_t num_batches, int64_t num_indices,
                       int64_t num_segments, int64_t instance_size, const K* indices, const T* in,
                       T* out);
};

template<typename T, typename K>
void UnsortedBatchSegmentSumKernelUtil<DeviceType::kGPU, T, K>::Dispatch(
    DeviceCtx* ctx, int64_t num_batches, int64_t num_indices, int64_t num_segments,
    int64_t instance_size, const K* indices, const T* in, T* out) {
  const size_t in_batch_size = num_indices * instance_size;
  const size_t out_batch_size = instance_size * num_segments;
  const size_t out_batch_size_in_bytes = out_batch_size * sizeof(T);
  if (num_batches >= kImplBatchWiseMinNumBatches
      && out_batch_size_in_bytes <= kImplBatchWiseMaxSharedBufSizeInBytes
      && in_batch_size >= kImplBatchWiseMinInBatchSize) {
    const int32_t thread_num = ImplBatchWiseGetThreadNum(in_batch_size);
    const int32_t block_num = ImplBatchWiseGetBlockNum(num_batches);
    ImplBatchWise<T, K><<<block_num, thread_num, out_batch_size_in_bytes, ctx->cuda_stream()>>>(
        num_batches, num_indices, num_segments, instance_size, indices, in, out);
  } else {
    const int64_t elem_cnt = num_batches * num_indices * instance_size;
    Memset<DeviceType::kGPU>(ctx, out, 0, num_batches * num_segments * instance_size * sizeof(T));
    ImplNaive<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, num_indices, num_segments, instance_size, indices, in, out);
  }
}

#define INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_GPU(data_type_pair, index_type_pair) \
  template struct UnsortedBatchSegmentSumKernelUtil<                                            \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
