#include "oneflow/core/kernel/reduce_sum_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");

  const Shape out_shape_kept_dims(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()));
  const Shape& in_shape = in_blob->shape();
  if (device_type == DeviceType::kCPU && out_shape_kept_dims.At(0) == in_shape.At(0)
      && out_shape_kept_dims.Count(1) == 1) {
    const int64_t n = in_shape.At(0);
    const int64_t m = in_shape.Count(1);
    const T* in_ptr = in_blob->dptr<T>();
    T* out_ptr = out_blob->mut_dptr<T>();
    FOR_RANGE(int64_t, i, 0, n) {
      out_ptr[i] = ZeroVal<T>::value;
      FOR_RANGE(int64_t, j, 0, m) { out_ptr[i] += in_ptr[i * m + j]; }
    }
  } else {
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx.device_ctx,
        XpuVarNdarray<T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                         out_blob->mut_dptr<T>()),
        XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
        XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  }
}
template<DeviceType device_type, typename T>
void ReduceSumKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  NdarrayUtil<device_type, T>::BroadcastTo(
      ctx.device_ctx, XpuVarNdarray<T>(in_diff_blob, in_diff_blob->shape().NumAxes()),
      XpuVarNdarray<const T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                             out_diff_blob->dptr<T>()));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceSumConf, ReduceSumKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
