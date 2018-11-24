#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

namespace{

  template<typename T>
  __global__ void TransInblob(const int64_t elem_cnt,const int64_t* dims,
                              const Shape& inshape,const Shape& outshape,
                              const Blob* in_blob, Blob* out_blob) {
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      int32_t count = i; 
      for(int32_t d = 0; d < inshape.NumAxes(); d++ ){
        dims[d] = count / outshape.Count(d);
        if(dims[d] >= outshape.At(d)){
          out_dptr[i] = ZeroVal<T>::value;
          return;
        }
        in_dptr += dims[d] * outshape.Count(d);
        count -= dims[d];
      }
      out_dptr[i] = *in_dptr;  
    }
  }
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingForward(const KernelCtx& kernel_ctx,
                                                        const PoolingCtx& pooling_ctx,
                                                        const Blob* in_blob, Blob* out_blob) const {
  if (!Global<JobDesc>::Get()->caffe_pad_head_more()){
    Blob* in_t_blob = BnInOp2Blob("in_t");
    TransformInBlob(kernel_ctx, in_blob, in_t_blob);
    CudaCheck(cudnnPoolingForward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), OnePtr<T>::value,
      pooling_ctx.cudnn_in_tensor_desc(), in_t_blob->dptr(), ZeroPtr<T>::value,
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->mut_dptr()));
  }else{
    CudaCheck(cudnnPoolingForward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), OnePtr<T>::value,
      pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(), ZeroPtr<T>::value,
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->mut_dptr()));
  }                                                        
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                                         const PoolingCtx& pooling_ctx,
                                                         const Blob* out_diff_blob,
                                                         const Blob* out_blob, const Blob* in_blob,
                                                         Blob* in_diff_blob) const {
  CudaCheck(cudnnPoolingBackward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), OnePtr<T>::value,
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->dptr(), pooling_ctx.cudnn_out_tensor_desc(),
      out_diff_blob->dptr(), pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(), ZeroPtr<T>::value,
      pooling_ctx.cudnn_in_tensor_desc(), in_diff_blob->mut_dptr()));
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::TransformInBlob(const KernelCtx& kernel_ctx,
                                                         const Blob* in_blob, 
                                                         Blob* out_blob) const {
  const int64_t elem_cnt = out_blob->shape().elem_cnt();
  const Shape& inshape = in_blob->shape();
  const Shape& outshape = out_blob->shape();
  int64_t dims[inshape.NumAxes()];
  TransInblob<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                kernel_ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, dims, inshape, outshape, in_blob->dptr<T>(), out_blob->mut_dptr<T>());                                  
}

#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class PoolingKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
