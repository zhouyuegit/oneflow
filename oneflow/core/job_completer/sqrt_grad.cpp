#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sqrt_conf());
  const SqrtOpConf& conf = op.op_conf().sqrt_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf sqrt_grad_op;
    sqrt_grad_op.set_name(op.op_name() + "_grad");
    SqrtGradOpConf* sqrt_grad_op_conf = sqrt_grad_op.mutable_sqrt_grad_conf();
    sqrt_grad_op_conf->set_out(GenLogicalBlobName(op.BninOp2Lbi("out")));
    sqrt_grad_op_conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    sqrt_grad_op_conf->set_in_diff("in_diff");
    op_confs->push_back(sqrt_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(sqrt_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("in_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSqrtConf, &GenerateBackwardOpConf);

}  // namespace oneflow
