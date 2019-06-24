#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sqrt_conf());
  const SqrtOpConf& conf = op.op_conf().sqrt_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf square_op;
    square_op.set_name(op.op_name() + "_grad");
    SquareOpConf* square_op_conf = square_op.mutable_square_conf();
    square_op_conf->set_out("out");
    square_op_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    op_confs->push_back(square_op);
    DiffLbi4BnInOp("in")->set_op_name(square_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSqrtConf, &GenerateBackwardOpConf);

}  // namespace oneflow
