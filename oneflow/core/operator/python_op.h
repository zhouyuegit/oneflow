#ifndef ONEFLOW_CORE_OPERATOR_PYTHON_OP_H_
#define ONEFLOW_CORE_OPERATOR_PYTHON_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct PythonOpCtx : public OpContext {
  int32_t axis;
  int32_t dims;
  int64_t transpose_rows;
  int64_t transpose_cols;
  bool need_transpose;
};

class PythonOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PythonOp);
  PythonOp() = default;
  ~PythonOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*, size_t* buf_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  PythonOpCtx* NewPythonOpCtx(const Shape& in_shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PYTHON_OP_H_
