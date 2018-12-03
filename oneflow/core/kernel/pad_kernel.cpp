#include "oneflow/core/kernel/pad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PadKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  Blob* padding_left_bound_blob = BnInOp2Blob("padding_left_bound");
  Blob* padding_right_bound_blob = BnInOp2Blob("padding_right_bound");
  Blob* inshape_count_blob = BnInOp2Blob("inshape_count");
  Blob* outshape_count_blob = BnInOp2Blob("outshape_count");

  LOG(INFO) << "before pad";
  PadKernelUtil<device_type, T>::Forward(ctx, 
              outshape_count_blob->mut_dptr<int32_t>(), inshape_count_blob->mut_dptr<int32_t>(),
              padding_left_bound_blob->mut_dptr<int32_t>(), padding_right_bound_blob->mut_dptr<int32_t>(),  
              in_blob, out_blob);
  LOG(INFO) << "after pad";
}

template<DeviceType device_type, typename T>
void PadKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  Blob* padding_left_bound_blob = BnInOp2Blob("padding_left_bound");
  Blob* padding_right_bound_blob = BnInOp2Blob("padding_right_bound");
  Blob* inshape_count_blob = BnInOp2Blob("inshape_count");
  Blob* outshape_count_blob = BnInOp2Blob("outshape_count");

  LOG(INFO) << "before pad diff";
  PadKernelUtil<device_type, T>::Backward(ctx, 
              outshape_count_blob->dptr<int32_t>(), inshape_count_blob->dptr<int32_t>(),
              padding_left_bound_blob->dptr<int32_t>(), padding_right_bound_blob->dptr<int32_t>(), 
              in_diff_blob, out_diff_blob);
  LOG(INFO) << "after pad diff";
}

template<typename T>
struct PadKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, int32_t* outshape_count, int32_t* inshape_count,
                      int32_t* padding_left_bound, int32_t* padding_right_bound, 
                      const Blob* in_blob, Blob* out_blob) {
    const Shape& outshape = out_blob->shape();
    const Shape& inshape = in_blob->shape();
    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    int64_t num_axes = outshape.NumAxes();

    const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();
    const PbRf<int32_t>& padding_after = ctx.kernel_conf().padding_after();

    for(int64_t i = 0; i < num_axes; i++){
      padding_left_bound[i] = padding_before.Get(i);
      padding_right_bound[i] = padding_after.Get(i) + inshape.At(i) - 1;
      outshape_count[i] = outshape.Count(i + 1);
      inshape_count[i] = inshape.Count(i + 1);
    }

    const T* in_dptr = in_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();

    for(size_t i = 0; i < elem_cnt ; i++ ){
      int64_t offset = i;
      int64_t index = 0; 
      for(size_t d = 0; d < num_axes; d++){
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

  static void Backward(const KernelCtx& ctx, const int32_t* outshape_count, const int32_t* inshape_count,
                      const int32_t* padding_left_bound, const int32_t* padding_right_bound,
                      Blob* in_diff_blob, const Blob* out_diff_blob) {
    const int64_t elem_cnt = out_diff_blob->shape().elem_cnt();
    int64_t num_axes = out_diff_blob->shape().NumAxes();

    T* in_diff_dptr = in_diff_blob->mut_dptr<T>();
    const T* out_diff_dptr = out_diff_blob->dptr<T>();

    for(int64_t i = 0; i < elem_cnt ; i++ ){
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
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPadConf, PadKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
