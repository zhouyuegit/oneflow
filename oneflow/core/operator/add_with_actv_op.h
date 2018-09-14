#ifndef ONEFLOW_CORE_OPERATOR_ADD_WITH_ACTV_OP_H_
#define ONEFLOW_CORE_OPERATOR_ADD_WITH_ACTV_OP_H_

#include "oneflow/core/operator/add_op.h"

namespace oneflow {

class AddWithActvOp final : public AddOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddWithActvOp);
  AddWithActvOp() = default;
  ~AddWithActvOp() = default;

  void VirtualInitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADD_WITH_ACTV_OP_H_
