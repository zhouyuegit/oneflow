#include "oneflow/core/kernel/pad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PadKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t elem_cnt = out_blob->shape().elem_cnt();
  const Shape& outshape = out_blob->shape();
  const Shape& inshape = in_blob->shape();
  int64_t num_axes = outshape.NumAxes();
  int64_t outshape_at[num_axes];
  int64_t inshape_at[num_axes];
  int64_t outshape_count[num_axes];
  int64_t inshape_count[num_axes];
  for(int64_t i = 0; i < num_axes; i++){
    outshape_at[i] = outshape.At(i);
    inshape_at[i] = inshape.At(i);
    outshape_count[i] = outshape.Count(i + 1);
    inshape_count[i] = inshape.Count(i + 1);
  }
  LOG(INFO) << "before pad";
  PadKernelUtil<device_type, T>::Forward(ctx, elem_cnt, num_axes, outshape_count, outshape_at,
              inshape_count, inshape_at, in_blob->dptr<T>(), out_blob->mut_dptr<T>());
  LOG(INFO) << "after pad";
}

template<typename T>
struct PadKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const int64_t elem_cnt, const int64_t num_axes,
                      const int64_t* outshape_count,const int64_t* outshape_at,
                      const int64_t* inshape_count,const int64_t* inshape_at,
                      const T* in_dptr, T* out_dptr) {
    for(int64_t i = 0; i < elem_cnt ; i++ ){
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
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPadConf, PadKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
