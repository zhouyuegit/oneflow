#ifndef ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MultipleGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultipleGatherOp);
  MultipleGatherOp() = default;
  ~MultipleGatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
  void InferIsModelBlob4OutputBlobs(
      std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_
