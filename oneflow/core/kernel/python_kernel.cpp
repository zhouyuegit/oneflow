#include "oneflow/core/kernel/python_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include <pybind11/embed.h>

namespace oneflow {

template<typename T>
void PythonKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conf = this->kernel_conf().python_conf();
  const std::string module_name = this->op_conf().python_conf().module_name();
  LOG(INFO) << "pybind11::gil_scoped_acquire acquire";
  pybind11::gil_scoped_acquire acquire;
  pybind11::module py_mod = pybind11::module::import(module_name.c_str());
  pybind11::object _ = py_mod.attr("forward_data_content")(1, 2);
  pybind11::gil_scoped_release release;
  LOG(INFO) << "pybind11::gil_scoped_release release";
  const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    Memcpy<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_memory_ptr(), in_blob->memory_ptr(),
                             in_blob->TotalByteSize());
  }
}

template<typename T>
void PythonKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  FOR_RANGE(size_t, i, 0, this->op_attribute().input_diff_bns().size()) {
    Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(i));
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kPythonConf, PythonKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
