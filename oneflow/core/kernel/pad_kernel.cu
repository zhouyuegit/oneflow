#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pad_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow{
namespace{

template<typename T>
__global__ void PadOneAfter(const int64_t elem_cnt, const int64_t num_axes,
                            const int64_t* outshape_count,const int64_t* outshape_at,
                            const int64_t* inshape_count,const int64_t* inshape_at,
                            const T* in_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = i;
    int64_t index = 0; 
    for(int64_t d = 0; d < num_axes; d++){
      int64_t dim = offset / outshape_count[d];
      // if this dim need padding
      if(dim >= inshape_at[d]){
        out_dptr[i] = ZeroVal<T>::value;
        break;
      }
      index += dim * inshape_count[d];
      offset -= dim * outshape_count[d];
      if(offset == 0){out_dptr[i] = in_dptr[index];}
    }
  }
}

}// namespace 

template<typename T>
struct PadKernelUtil<DeviceType::kGPU, T>{
  static void Forward(const KernelCtx& ctx, const int64_t elem_cnt, const int64_t num_axes,
                      const int64_t* outshape_count,const int64_t* outshape_at,
                      const int64_t* inshape_count,const int64_t* inshape_at,
                      const T* in_dptr, T* out_dptr){
  int64_t size = num_axes * sizeof(int64_t);
  int64_t* d_outshape_count;
  int64_t* d_outshape_at;
  int64_t* d_inshape_at;
  int64_t* d_inshape_count;

  CudaCheck(cudaMalloc((void**)&d_outshape_count, size));
  CudaCheck(cudaMalloc((void**)&d_outshape_at, size));
  CudaCheck(cudaMalloc((void**)&d_inshape_count, size));
  CudaCheck(cudaMalloc((void**)&d_inshape_at, size));

  CudaCheck(cudaMemcpy(d_outshape_count, outshape_count, size, cudaMemcpyHostToDevice));
  CudaCheck(cudaMemcpy(d_outshape_at, outshape_at, size, cudaMemcpyHostToDevice));
  CudaCheck(cudaMemcpy(d_inshape_count, inshape_count, size, cudaMemcpyHostToDevice));
  CudaCheck(cudaMemcpy(d_inshape_at, inshape_at, size, cudaMemcpyHostToDevice));


  PadOneAfter<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
              ctx.device_ctx->cuda_stream()>>>(elem_cnt, num_axes, d_outshape_count, d_outshape_at,
              d_inshape_count, d_inshape_at, in_dptr, out_dptr);
  
  CudaCheck(cudaFree(d_outshape_count));
  CudaCheck(cudaFree(d_outshape_at));
  CudaCheck(cudaFree(d_inshape_count));
  CudaCheck(cudaFree(d_inshape_at));

  }
};

#define INSTANTIATE_PAD_KERNEL_UTIL(type_cpp, type_proto) \
  template class PadKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}// namespace oneflow