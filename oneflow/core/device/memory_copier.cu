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
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

template<int32_t NDIMS>
struct Int32Array {
  int32_t val[NDIMS];
};

template<int32_t NDIMS, typename T>
__global__ void CopyNDGpu(const int n, T* dst, const T* src,
                          NdIndexOffsetHelper<int64_t, NDIMS> dst_helper,
                          NdIndexOffsetHelper<int64_t, NDIMS> src_helper,
                          NdIndexOffsetHelper<int64_t, NDIMS> copy_helper,
                          Int32Array<NDIMS> dst_pos, Int32Array<NDIMS> src_pos) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t copy_idx[NDIMS];
    int64_t src_idx[NDIMS];
    int64_t dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
#pragma unroll
    for (int64_t j = 0; j < NDIMS; j++) {
      src_idx[j] = src_pos.val[j] + copy_idx[j];
      dst_idx[j] = dst_pos.val[j] + copy_idx[j];
    }
    const int64_t src_offset = src_helper.NdIndexToOffset(src_idx);
    const int64_t dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    dst[dst_offset] = src[src_offset];
  }
}

size_t GetPackSize(const MemoryCopyNdDesc& desc, const void* dst, const void* src) {
  const int64_t mask = desc.src_shape.dim_vec().back() | desc.dst_shape.dim_vec().back()
                       | desc.extent.dim_vec().back() | desc.src_pos.dim_vec().back()
                       | desc.dst_pos.dim_vec().back() | reinterpret_cast<uintptr_t>(dst)
                       | reinterpret_cast<uintptr_t>(src);
  if ((mask & 0xF) == 0) {
    return 16;
  } else if ((mask & 0x7) == 0) {
    return 8;
  } else if ((mask & 0x3) == 0) {
    return 4;
  } else if ((mask & 0x1) == 0) {
    return 2;
  } else {
    return 1;
  }
}

}  // namespace

template<int32_t NDIMS>
void CopyNDGpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  CHECK_EQ(desc.dst_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.dst_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.extent.NumAxes(), NDIMS);

  const size_t pack_size = GetPackSize(desc, dst, src);

  DimVector src_shape_dim_vec = desc.src_shape.dim_vec();
  DimVector dst_shape_dim_vec = desc.dst_shape.dim_vec();
  DimVector extent_dim_vec = desc.extent.dim_vec();
  DimVector src_pos_dim_vec = desc.src_pos.dim_vec();
  DimVector dst_pos_dim_vec = desc.dst_pos.dim_vec();

  src_shape_dim_vec.back() /= pack_size;
  dst_shape_dim_vec.back() /= pack_size;
  extent_dim_vec.back() /= pack_size;
  src_pos_dim_vec.back() /= pack_size;
  dst_pos_dim_vec.back() /= pack_size;

  NdIndexOffsetHelper<int64_t, NDIMS> src_helper(src_shape_dim_vec.data());
  NdIndexOffsetHelper<int64_t, NDIMS> dst_helper(dst_shape_dim_vec.data());
  NdIndexOffsetHelper<int64_t, NDIMS> copy_helper(extent_dim_vec.data());
  Int32Array<NDIMS> src_pos;
  Int32Array<NDIMS> dst_pos;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    dst_pos.val[i] = dst_pos_dim_vec.at(i);
    src_pos.val[i] = src_pos_dim_vec.at(i);
  }
  const int64_t elem_cnt = desc.extent.elem_cnt() / pack_size;
  if (pack_size == 1) {
    CopyNDGpu<NDIMS, uint8_t>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, reinterpret_cast<uint8_t*>(dst), reinterpret_cast<const uint8_t*>(src),
            dst_helper, src_helper, copy_helper, dst_pos, src_pos);
  } else if (pack_size == 2) {
    CopyNDGpu<NDIMS, uint16_t>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, reinterpret_cast<uint16_t*>(dst), reinterpret_cast<const uint16_t*>(src),
            dst_helper, src_helper, copy_helper, dst_pos, src_pos);
  } else if (pack_size == 4) {
    CopyNDGpu<NDIMS, uint32_t>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, reinterpret_cast<uint32_t*>(dst), reinterpret_cast<const uint32_t*>(src),
            dst_helper, src_helper, copy_helper, dst_pos, src_pos);
  } else if (pack_size == 8) {
    CopyNDGpu<NDIMS, uint64_t>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, reinterpret_cast<uint64_t*>(dst), reinterpret_cast<const uint64_t*>(src),
            dst_helper, src_helper, copy_helper, dst_pos, src_pos);
  } else if (pack_size == 16) {
    static_assert(sizeof(uint4) == 16, "");
    CopyNDGpu<NDIMS, uint4>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, reinterpret_cast<uint4*>(dst), reinterpret_cast<const uint4*>(src),
            dst_helper, src_helper, copy_helper, dst_pos, src_pos);
  } else {
    UNIMPLEMENTED();
  }
}

#define SPECIALIZE_COPY_ND_GPU_IMPL(NDIMS)                                        \
  template void CopyNDGpuImpl<NDIMS>(DeviceCtx * ctx, void* dst, const void* src, \
                                     const MemoryCopyNdDesc& desc);
SPECIALIZE_COPY_ND_GPU_IMPL(2)
SPECIALIZE_COPY_ND_GPU_IMPL(3)
SPECIALIZE_COPY_ND_GPU_IMPL(4)
SPECIALIZE_COPY_ND_GPU_IMPL(5)
SPECIALIZE_COPY_ND_GPU_IMPL(6)

}  // namespace oneflow
