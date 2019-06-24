#include "oneflow/core/kernel/prelu_grad_kernel.h"
#include "oneflow/core/kernel/prelu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* alpha_diff_blob = BnInOp2Blob("alpha_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  Memset<device_type>(ctx.device_ctx, alpha_diff_blob->mut_dptr<T>(), 0,
                      alpha_diff_blob->ByteSizeOfDataContentField());
  PReluKernelUtil<device_type, T>::Backward(
      ctx, this->op_conf().prelu_grad_conf(), this->kernel_conf().prelu_grad_conf().perm(), BnInOp2Blob("in"),
      BnInOp2Blob("alpha"), BnInOp2Blob("out_diff"), BnInOp2Blob("fw_buf"), in_diff_blob,
      alpha_diff_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPreluGradConf, PReluGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
