#ifndef ONEFLOW_CORE_OPERATOR_PAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_PAD_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

class PadOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadOp);
  PadOp() = default;
  ~PadOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  Shape GetOutShape(int64_t in_n, int64_t in_c, int64_t dims,
                    std::string data_format, const std::vector<int64_t>& out) const;
 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PAD_OP_H_