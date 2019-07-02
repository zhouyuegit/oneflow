#include "oneflow/core/kernel/reduce_max_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceMaxKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  size_t count = in_blob->shape().elem_cnt() / out_blob->shape().elem_cnt();
  NdarrayUtil<device_type, T>::ReduceMax(
      ctx.device_ctx,
      XpuVarNdarray<T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                       out_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
      XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
}

template<DeviceType device_type, typename T>
void ReduceMaxKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    TODO();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceMaxConf, ReduceMaxKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
