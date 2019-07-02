#include "oneflow/core/kernel/softmax_reduce_max_stage0_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SoftmaxReduceMaxStage0Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* max_blob = BnInOp2Blob("max");
  Blob* max_count_blob = BnInOp2Blob("max_count");
  Blob* mask_blob = BnInOp2Blob("mask");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  Blob* fw_tmp_int_blob = BnInOp2Blob("fw_tmp_int");
  size_t count = in_blob->shape().elem_cnt() / max_blob->shape().elem_cnt();
  NdarrayUtil<device_type, T>::ReduceMax(
      ctx.device_ctx,
      XpuVarNdarray<T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                       max_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
      XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  SoftmaxReduceMaxStage0KernelUtil<device_type, T>::SetMask(ctx.device_ctx, in_blob->shape().elem_cnt(), count, in_blob->dptr<T>(), max_blob->dptr<T>(), mask_blob->mut_dptr<int32_t>());
  NdarrayUtil<device_type, int32_t>::ReduceSum(
        ctx.device_ctx, XpuVarNdarray<int32_t>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()), max_count_blob->mut_dptr<int32_t>()),
        XpuVarNdarray<const int32_t>(const_cast<const Blob*>(mask_blob), in_blob->shape().NumAxes()), XpuVarNdarray<int32_t>(fw_tmp_int_blob, in_blob->shape().NumAxes()));

}

template<DeviceType device_type, typename T>
void SoftmaxReduceMaxStage0Kernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    Blob* max_diff_blob = BnInOp2Blob(GenDiffBn("max"));
    Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
    Blob* max_count_blob = BnInOp2Blob("max_count");
    Blob* mask_blob = BnInOp2Blob("mask");
    size_t count = in_diff_blob->shape().elem_cnt() / max_diff_blob->shape().elem_cnt();
    SoftmaxReduceMaxStage0KernelUtil<device_type, T>::SetWithMask(ctx.device_ctx, in_diff_blob->shape().elem_cnt(), count, max_diff_blob->dptr<T>(), mask_blob->dptr<int32_t>(), max_count_blob->dptr<int32_t>(), in_diff_blob->mut_dptr<T>());
}


template<typename T>
struct SoftmaxReduceMaxStage0KernelUtil<DeviceType::kCPU, T> {
  static void SetMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* in, const T* max, int32_t* mask) {
    FOR_RANGE(int64_t, i, 0, n){ 
       mask[i] = (in[i] == max[i / count]) ? 1 : 0;
    }
  }

  static void SetWithMask(DeviceCtx* ctx, const int64_t n, const int64_t count, const T* max_diff, const int32_t* mask, const int32_t* max_count,
                                T* in_diff) {
    FOR_RANGE(int64_t, i, 0, n){ 
      in_diff[i] = max_diff[i / count] * mask[i] / max_count[i / count] ;
    }
  }
};
#define INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxReduceMaxStage0KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_REDUCE_MAX_STAGE0_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxReduceMaxStage0Conf, SoftmaxReduceMaxStage0Kernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
