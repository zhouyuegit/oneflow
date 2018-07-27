#include "oneflow/core/operator/python_op.h"

namespace oneflow {

void PythonOp::InitFromOpConf() {
  CHECK(op_conf().has_python_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("python_num");
  EnrollDataTmpBn("transpose_in");
  EnrollDataTmpBn("transpose_out");
  EnrollDataTmpBn("transpose_out_diff");
}

const PbMessage& PythonOp::GetCustomizedConf() const { return op_conf().python_conf(); }

void PythonOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx, size_t* buf_size,
                              std::function<void(OpContext*)> EnrollOpCtx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  PythonOpCtx* op_ctx = NewPythonOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);
}

void PythonOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  PythonKernelConf* conf = kernel_conf->mutable_python_conf();
  const PythonOpCtx* python_ctx = static_cast<const PythonOpCtx*>(op_ctx);
}

PythonOpCtx* PythonOp::NewPythonOpCtx(const Shape& in_shape) const {
  PythonOpCtx* op_ctx = new PythonOpCtx();
  return op_ctx;
}

REGISTER_OP(OperatorConf::kPythonConf, PythonOp);

}  // namespace oneflow
