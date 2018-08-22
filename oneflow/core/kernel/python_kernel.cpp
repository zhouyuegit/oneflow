#include "oneflow/core/kernel/python_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <numpy/arrayobject.h>

namespace oneflow {

int32_t OneFlow2NumpyType(DataType data_type) {
  static std::map<DataType, int32_t> oneflow2numpy_map{{DataType::kFloat, NPY_FLOAT},
                                                       {DataType::kDouble, NPY_DOUBLE},
                                                       {DataType::kInt8, NPY_INT8},
                                                       {DataType::kInt32, NPY_INT}};
  const auto found_it = oneflow2numpy_map.find(data_type);
  if (found_it == oneflow2numpy_map.end()) {
    UNIMPLEMENTED();
  } else {
    return found_it->second;
  }
}

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

    std::vector<npy_intp> numpy_dims;
    for (const auto dim : in_blob->shape().dim_vec()) { numpy_dims.push_back(dim); }
    pybind11::object in_array = pybind11::reinterpret_steal<pybind11::object>(
        PyArray_SimpleNewFromData(in_blob->shape().NumAxes(), numpy_dims.data(),
                                  OneFlow2NumpyType(in_blob->data_type()), in_blob->mut_dptr()));

    in_array_vec.push_back(in_array);
  }

  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);

    std::vector<npy_intp> numpy_dims;
    for (const auto dim : out_blob->shape().dim_vec()) { numpy_dims.push_back(dim); }
    pybind11::object out_array = pybind11::reinterpret_steal<pybind11::object>(
        PyArray_SimpleNewFromData(out_blob->shape().NumAxes(), numpy_dims.data(),
                                  OneFlow2NumpyType(out_blob->data_type()), out_blob->mut_dptr()));

    // for debug workaround
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
    out_array_vec.push_back(out_array);
  }

  pybind11::object _ = py_mod.attr("forward_data_content")(in_array_vec[0], out_array_vec[0]);
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
