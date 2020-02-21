#ifndef ONEFLOW_CORE_OPERATOR_CWISE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CWISE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CWiseOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CWiseOp);
  CWiseOp() = default;
  virtual ~CWiseOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferTmpBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualInitFromOpConf() { UNIMPLEMENTED(); }

  virtual Maybe<void> VirtualInferOutBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    return Maybe<void>::Ok();
  }
  virtual Maybe<void> VirtualInferTmpBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    return Maybe<void>::Ok();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CWISE_OP_H_
