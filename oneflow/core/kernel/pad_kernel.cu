#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pad_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow{
namespace{

template<typename T>
__global__ void PadForward(const int64_t elem_cnt, const int64_t num_axes,
                            const int32_t* outshape_count,const int32_t* inshape_count,
                            const int32_t* padding_left_bound,const int32_t* padding_right_bound,
                            const T* in_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = i;
    int64_t index = 0; 
    for(int64_t d = 0; d < num_axes; d++){
      int64_t dim = offset / outshape_count[d];
      // if this dim need padding
      if(dim >= padding_right_bound[d] || dim < padding_left_bound[d]){
        out_dptr[i] = ZeroVal<T>::value;
        break;
      }
      index += (dim - padding_left_bound[d]) * inshape_count[d];
      offset -= dim * outshape_count[d];
      if(offset == 0){out_dptr[i] = in_dptr[index];}
    }
  }
}

template<typename T>
__global__ void PadBackward(const int64_t elem_cnt, const int64_t num_axes,
                            const int32_t* outshape_count,const int32_t* inshape_count,
                            const int32_t* padding_left_bound,const int32_t* padding_right_bound,
                            T* in_diff_dptr, const T* out_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = i;
    int64_t index = 0; 
    for(int64_t d = 0; d < num_axes; d++){
      int64_t dim = offset / outshape_count[d];
      // if this dim need padding
      if(dim >= padding_right_bound[d] || dim < padding_left_bound[d]){break;}
      index += (dim - padding_left_bound[d]) * inshape_count[d];
      offset -= dim * outshape_count[d];
      if(offset == 0){in_diff_dptr[index] = out_diff_dptr[i];}
    }
  }
}

}// namespace 

template<typename T>
struct PadKernelUtil<DeviceType::kGPU, T>{
  static void Forward(const KernelCtx& ctx, 
                      const PbRf<int32_t>& padding_before, const PbRf<int32_t>& padding_after,
                      int32_t* outshape_count, int32_t* inshape_count,
                      int32_t* padding_left_bound, int32_t* padding_right_bound, 
                      const Blob* in_blob, Blob* out_blob){

    const Shape& outshape = out_blob->shape();
    const Shape& inshape = in_blob->shape();
    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    int64_t num_axes = outshape.NumAxes();

    int32_t size = num_axes * sizeof(int32_t);
    int32_t h_outshape_count[num_axes];
    int32_t h_inshape_count[num_axes];
    int32_t h_padding_left_bound[num_axes];
    int32_t h_padding_right_bound[num_axes]; 

    for(int64_t i = 0; i < num_axes; i++){
      h_padding_left_bound[i] = padding_before.Get(i);
      h_padding_right_bound[i] = padding_before.Get(i) + static_cast<int32_t>(inshape.At(i));
      h_outshape_count[i] = static_cast<int32_t>(outshape.Count(i + 1));
      h_inshape_count[i] = static_cast<int32_t>(inshape.Count(i + 1));
    }

    CudaCheck(cudaMemcpy(outshape_count, h_outshape_count, size, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(inshape_count, h_inshape_count, size, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(padding_left_bound, h_padding_left_bound, size, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(padding_right_bound, h_padding_right_bound, size, cudaMemcpyHostToDevice));

    PadForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                ctx.device_ctx->cuda_stream()>>>(elem_cnt, num_axes, outshape_count, inshape_count,
                padding_left_bound, padding_right_bound, in_blob->dptr<T>(), out_blob->mut_dptr<T>());
  }

  static void Backward(const KernelCtx& ctx, const int32_t* outshape_count, const int32_t* inshape_count,
                       const int32_t* padding_left_bound, const int32_t* padding_right_bound,
                       Blob* in_diff_blob, const Blob* out_diff_blob) {
    const int64_t elem_cnt = out_diff_blob->shape().elem_cnt();
    int64_t num_axes = out_diff_blob->shape().NumAxes();
    
    PadBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                ctx.device_ctx->cuda_stream()>>>(elem_cnt, num_axes, outshape_count, inshape_count,
                padding_left_bound, padding_right_bound, in_diff_blob->mut_dptr<T>(), out_diff_blob->dptr<T>());
  }
};

#define INSTANTIATE_PAD_KERNEL_UTIL(type_cpp, type_proto) \
  template class PadKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}// namespace oneflow