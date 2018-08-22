#include "oneflow/core/kernel/python_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace oneflow {

template<typename T>
void PythonKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  pybind11::gil_scoped_acquire acquire;
  const std::string module_name = this->op_conf().python_conf().module_name();

  pybind11::module py_mod = pybind11::module::import(module_name.c_str());

  std::vector<pybind11::object> in_array_vec;
  in_array_vec.reserve(this->op_attribute().input_bns_size());
  std::vector<pybind11::object> out_array_vec;
  out_array_vec.reserve(this->op_attribute().output_bns_size());

  for (const std::string& ibn : this->op_attribute().input_bns()) {
    Blob* in_blob = BnInOp2Blob(ibn);
    pybind11::object in_array;
    in_array_vec.push_back(in_array);
  }

  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);

    // for debug workaround
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
    pybind11::object out_array;
    out_array_vec.push_back(out_array);
  }

  pybind11::object _ = py_mod.attr("forward_data_content")(in_array_vec, out_array_vec);
  pybind11::gil_scoped_release release;
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
