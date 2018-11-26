#include "oneflow/core/kernel/pad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PadKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

}
  KernelUtil<device_type, T>::Pad(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

void PadOneAfter(const int64_t elem_cnt, const int64_t num_axes,
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

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPadConf, PadKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow