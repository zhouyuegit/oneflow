#include "oneflow/core/kernel/log_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void LogKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  LogKernelUtil<device_type, T>::Log(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                   in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void LogKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  KernelUtil<device_type, T>::Div(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                  out_diff_blob->dptr<T>(), in_blob->dptr<T>(),
                                  in_diff_blob->mut_dptr<T>());
}


template<typename T>
struct LogKernelUtil<DeviceType::kCPU, T> {
  static void Log(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = SafeLog(x[i]); }
  }
};
#define INSTANTIATE_LOG_KERNEL_UTIL(type_cpp, type_proto) \
  template struct LogKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LOG_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)


ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLogConf, LogKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
