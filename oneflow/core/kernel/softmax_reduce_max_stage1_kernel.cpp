#include "oneflow/core/kernel/softmax_reduce_max_stage1_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SoftmaxReduceMaxStage1Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");  
  Blob* mask_blob = BnInOp2Blob("mask");
  Blob* data_tmp_blob = BnInOp2Blob("data_tmp");
  size_t count = in_blob->shape().elem_cnt() / out_blob->shape().elem_cnt();
  NdarrayUtil<device_type, T>::ReduceMax(
      ctx.device_ctx,
      XpuVarNdarray<T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                       out_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
      XpuVarNdarray<T>(data_tmp_blob, in_blob->shape().NumAxes()));
  SoftmaxReduceMaxStage1KernelUtil<device_type, T>::SetMask(ctx.device_ctx, in_blob->shape().elem_cnt(), count, in_blob->dptr<T>(), out_blob->dptr<T>(), mask_blob->mut_dptr<int32_t>());

}

template<DeviceType device_type, typename T>
void SoftmaxReduceMaxStage1Kernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
    Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
    const Blob* max_count_blob = BnInOp2Blob("max_count");
    const Blob* mask_blob = BnInOp2Blob("mask");    
    Blob* max_count_with_mask_blob = BnInOp2Blob("max_count_with_mask");
    Blob* data_tmp_int_blob = BnInOp2Blob("data_tmp_int");
    Blob* global_max_count_blob = BnInOp2Blob("global_max_count");
    size_t count = in_diff_blob->shape().elem_cnt() / out_diff_blob->shape().elem_cnt();
    KernelUtil<device_type, int32_t>::Multiply(ctx.device_ctx, in_diff_blob->shape().elem_cnt(), max_count_blob->dptr<int32_t>(),
                                mask_blob->dptr<int32_t>(), max_count_with_mask_blob->mut_dptr<int32_t>());
    NdarrayUtil<device_type, int32_t>::ReduceSum(
        ctx.device_ctx, XpuVarNdarray<int32_t>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()), global_max_count_blob->mut_dptr<int32_t>()),
        XpuVarNdarray<const int32_t>(const_cast<const Blob*>(max_count_with_mask_blob), max_count_blob->shape().NumAxes()), XpuVarNdarray<int32_t>(data_tmp_int_blob, max_count_blob->shape().NumAxes()));

    SoftmaxReduceMaxStage1KernelUtil<device_type, T>::SetWithMask(ctx.device_ctx, in_diff_blob->shape().elem_cnt(), count, out_diff_blob->dptr<T>(), mask_blob->dptr<int32_t>(), max_count_blob->dptr<int32_t>(), global_max_count_blob->dptr<int32_t>(), in_diff_blob->mut_dptr<T>());
}


template<typename T>
struct SoftmaxReduceMaxStage1KernelUtil<DeviceType::kCPU, T> {
  static void SetMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* in, const T* out, int32_t* mask) {
    FOR_RANGE(int32_t, i, 0, n){ 
       mask[i] = (in[i] == out[i / count]) ? 1 : 0;
    }
  }

  static void SetWithMask(DeviceCtx* ctx, const int32_t n, const int32_t count, const T* out_diff, const int32_t* mask, const int32_t* max_count, const int32_t* global_max_count,
                                T* in_diff) {
    FOR_RANGE(int32_t, i, 0, n){ 
      in_diff[i] = out_diff[i / count] * mask[i] * max_count[i] / global_max_count[i / count];     
    }
  }
};

#define INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxReduceMaxStage1KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE1_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxReduceMaxStage1Conf, SoftmaxReduceMaxStage1Kernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
