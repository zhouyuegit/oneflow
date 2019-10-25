#include "oneflow/core/kernel/unique_kernel_util.h"
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

namespace oneflow {

namespace {

template<typename T>
struct Buffer final {
  T* ptr = nullptr;
  size_t size_in_bytes = 0;
};

int64_t SizeAlign(int64_t size) { return RoundUp(size, kCudaAlignSize); }

template<typename T>
int64_t GetTempBufferSize(int64_t n) {
  return SizeAlign(n * sizeof(T));
}

template<typename KEY, typename IDX>
int64_t GetCubSortTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  CudaCheck(cub::DeviceRadixSort::SortPairs<KEY, IDX>(nullptr, cub_sort_temp_store_size, nullptr,
                                                      nullptr, nullptr, nullptr, n));
  CHECK_GE(cub_sort_temp_store_size, 0);
  CHECK_LT(cub_sort_temp_store_size, GetMaxVal<int64_t>());
  return SizeAlign(static_cast<int64_t>(cub_sort_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubRleTempStorageSize(int64_t n) {
  size_t cub_rle_temp_store_size = 0;
  CudaCheck(cub::DeviceRunLengthEncode::Encode<KEY*, KEY*, IDX*, int64_t*>(
      nullptr, cub_rle_temp_store_size, nullptr, nullptr, nullptr, nullptr, n));
  CHECK_GE(cub_rle_temp_store_size, 0);
  CHECK_LT(cub_rle_temp_store_size, GetMaxVal<int64_t>());
  return SizeAlign(static_cast<int64_t>(cub_rle_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubScanTempStorageSize(int64_t n) {
  size_t cub_scan_temp_store_size = 0;
  CudaCheck(cub::DeviceScan::ExclusiveSum<IDX*, IDX*>(nullptr, cub_scan_temp_store_size, nullptr,
                                                      nullptr, n));
  CHECK_GE(cub_scan_temp_store_size, 0);
  CHECK_LT(cub_scan_temp_store_size, GetMaxVal<int64_t>());
  return SizeAlign(static_cast<int64_t>(cub_scan_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubTempStorageSize(int64_t n) {
  int64_t cub_temp_storage_size = 0;
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubSortTempStorageSize<KEY, IDX>(n));
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubRleTempStorageSize<KEY, IDX>(n));
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubScanTempStorageSize<KEY, IDX>(n));
  return cub_temp_storage_size;
}

template<typename T>
void AliasPtr(void* origin, int64_t* offset, Buffer<T>* buffer, int64_t size) {
  auto* ptr = reinterpret_cast<unsigned char*>(origin);
  if (buffer != nullptr) {
    buffer->ptr = reinterpret_cast<T*>(ptr + *offset);
    buffer->size_in_bytes = size;
  }
  *offset += size;
}

template<typename KEY, typename IDX>
void UniqueAliasWorkspace(DeviceCtx* ctx, int64_t n, void* workspace,
                          int64_t* workspace_size_in_bytes, Buffer<KEY>* cub_sort_keys_out,
                          Buffer<IDX>* cub_sort_values_out, Buffer<IDX>* cub_scan_d_out,
                          Buffer<IDX>* rle_decode_out, Buffer<void>* cub_temp_storage) {
  int64_t offset = 0;
  AliasPtr(workspace, &offset, cub_sort_keys_out, GetTempBufferSize<KEY>(n));
  AliasPtr(workspace, &offset, cub_sort_values_out, GetTempBufferSize<IDX>(n));
  AliasPtr(workspace, &offset, cub_scan_d_out, GetTempBufferSize<IDX>(n));
  AliasPtr(workspace, &offset, rle_decode_out, GetTempBufferSize<IDX>(n));
  AliasPtr(workspace, &offset, cub_temp_storage, GetCubTempStorageSize<KEY, IDX>(n));
  *workspace_size_in_bytes = offset;
}

template<typename IDX>
__global__ void IotaKernel(int64_t n, IDX* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = static_cast<IDX>(i); }
}

const int32_t kRleDecodeBlockThreshold = 64;

template<typename IDX>
__global__ void RleDecodeByThreadKernel(const IDX* n, IDX* offsets, IDX* counts, IDX* out) {
  CUDA_1D_KERNEL_LOOP(i, *n) {
    const IDX offset = offsets[i];
    const IDX count = counts[i];
    if (count < kRleDecodeBlockThreshold) {
      for (IDX j = offset; j < offset + count; ++j) { out[j] = i; }
    }
  }
}

template<typename IDX>
__global__ void RleDecodeByBlockKernel(const IDX* n, IDX* offsets, IDX* counts, IDX* out) {
  for (int32_t bid = blockIdx.x; bid < *n; bid += gridDim.x) {
    const IDX offset = offsets[bid];
    const IDX count = counts[bid];
    if (count >= kRleDecodeBlockThreshold) {
      for (int32_t tid = offset + threadIdx.x; tid < offset + count; tid += blockDim.x) {
        out[tid] = bid;
      }
    }
  }
}

template<typename IDX>
__global__ void GatherOutIndexKernel(const int64_t n, const IDX* k, const IDX* v, IDX* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[k[i]] = v[i]; }
}

template<typename KEY, typename IDX>
__global__ void CheckKernel(const int64_t n, const KEY* in, const IDX* num_unique,
                            const KEY* unique_out, const IDX* idx_out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    IDX idx = idx_out[i];
    assert(idx < *num_unique);
    assert(unique_out[idx] == in[i]);
  }
}

}  // namespace

template<typename KEY, typename IDX>
struct UniqueKernelUtil<DeviceType::kGPU, KEY, IDX> {
  static void Unique(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes);
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes);
};

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::Unique(DeviceCtx* ctx, int64_t n, const KEY* in,
                                                          IDX* num_unique, KEY* unique_out,
                                                          IDX* idx_out, void* workspace,
                                                          int64_t workspace_size_in_bytes) {
  int64_t rt_workspace_size;
  IDX* cub_sort_values_in_ptr = idx_out;
  IDX* cub_rle_counts_out = idx_out;
  Buffer<KEY> cub_sort_keys_out;
  Buffer<IDX> cub_sort_values_out;
  Buffer<IDX> cub_scan_d_out;
  Buffer<IDX> rle_decode_out;
  Buffer<void> cub_temp_storage;
  UniqueAliasWorkspace<KEY, IDX>(ctx, n, workspace, &rt_workspace_size, &cub_sort_keys_out,
                                 &cub_sort_values_out, &cub_scan_d_out, &rle_decode_out,
                                 &cub_temp_storage);
  CHECK_LE(rt_workspace_size, workspace_size_in_bytes);
  IotaKernel<IDX><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, cub_sort_values_in_ptr);
  CudaCheck(cub::DeviceRadixSort::SortPairs<KEY, IDX>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, in, cub_sort_keys_out.ptr,
      cub_sort_values_in_ptr, cub_sort_values_out.ptr, n, 0, sizeof(KEY) * 8, ctx->cuda_stream()));
  CudaCheck(cub::DeviceRunLengthEncode::Encode<KEY*, KEY*, IDX*, IDX*>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, cub_sort_keys_out.ptr, unique_out,
      cub_rle_counts_out, num_unique, n, ctx->cuda_stream()));
  CudaCheck(cub::DeviceScan::ExclusiveSum<IDX*, IDX*>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, cub_rle_counts_out, cub_scan_d_out.ptr,
      n, ctx->cuda_stream()));
  RleDecodeByThreadKernel<IDX>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          num_unique, cub_scan_d_out.ptr, cub_rle_counts_out, rle_decode_out.ptr);
  RleDecodeByBlockKernel<IDX>
      <<<kCudaMaxBlocksNum, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          num_unique, cub_scan_d_out.ptr, cub_rle_counts_out, rle_decode_out.ptr);
  GatherOutIndexKernel<IDX>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, cub_sort_values_out.ptr, rle_decode_out.ptr, idx_out);
  CheckKernel<KEY, IDX>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, in, num_unique, unique_out, idx_out);
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueAliasWorkspace<KEY, IDX>(ctx, n, nullptr, workspace_size_in_bytes, nullptr, nullptr,
                                 nullptr, nullptr, nullptr);
}

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU(key_type_pair, idx_type_pair)              \
  template struct UniqueKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(key_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU, UNIQUE_KERNEL_KV_DATA_TYPE_SEQ,
                                 UNIQUE_KERNEL_KV_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU

}  // namespace oneflow
