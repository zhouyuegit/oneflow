#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"

namespace oneflow {

namespace {

constexpr int32_t kBatchMemcpyGpuNumThreadPerBlock = 256;

__global__ void DoCopy(const BatchMemcpyParams params) {
  const int32_t block_id = blockIdx.x;
  const int32_t thread_id = threadIdx.x;
  const int32_t num_thread = blockDim.x;
  uint64_t* bulk_dst = reinterpret_cast<uint64_t*>(params.dst[block_id]);
  const uint64_t* bulk_src = reinterpret_cast<const uint64_t*>(params.src[block_id]);
  const int32_t num_bulk = params.size[block_id] / sizeof(uint64_t);
  const int32_t num_byte = params.size[block_id] % sizeof(uint64_t);
  const int32_t bulk_size = num_bulk * sizeof(uint64_t);
  uint8_t* byte_dst = reinterpret_cast<uint8_t*>(params.dst[block_id]) + bulk_size;
  const uint8_t* byte_src = reinterpret_cast<const uint8_t*>(params.src[block_id]) + bulk_size;
  for (int32_t i = thread_id; i < num_bulk; i += num_thread) { bulk_dst[i] = bulk_src[i]; }
  if (thread_id < num_byte) { byte_dst[thread_id] = byte_src[thread_id]; }
}

}  // namespace

template<>
void BatchMemcpyKernelUtil<DeviceType::kGPU>::Copy(DeviceCtx* ctx,
                                                   const BatchMemcpyParams& params) {
  DoCopy<<<params.num_params, kBatchMemcpyGpuNumThreadPerBlock, 0, ctx->cuda_stream()>>>(params);
}

}  // namespace oneflow
