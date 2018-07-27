#include "oneflow/core/operator/python_op.h"

namespace oneflow {

void PythonOp::InitFromOpConf() {
  CHECK(op_conf().has_python_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& PythonOp::GetCustomizedConf() const { return op_conf().python_conf(); }

void PythonOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kPythonConf, PythonOp);

}  // namespace oneflow
