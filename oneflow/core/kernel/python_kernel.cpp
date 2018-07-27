#include "oneflow/core/kernel/python_kernel.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
void PythonKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  Blob* tmp_blob = BnInOp2Blob("python_num");
  auto conf = this->kernel_conf().python_conf();
}

template<typename T>
void PythonKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  const Blob* out_diff_blob = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
  Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(0));
  Blob* tmp_blob = BnInOp2Blob("python_num");
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kPythonConf, PythonKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
