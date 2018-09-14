#include "oneflow/core/operator/add_with_actv_op.h"

namespace oneflow {

void AddWithActvOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_with_actv_conf()); }
const PbMessage& AddWithActvOp::GetCustomizedConf() const { return op_conf().add_with_actv_conf(); }

REGISTER_OP(OperatorConf::kAddWithActvConf, AddWithActvOp);

}  // namespace oneflow
