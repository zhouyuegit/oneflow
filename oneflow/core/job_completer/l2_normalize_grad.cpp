#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_l2_normalize_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf l2_normalize_grad_op;
    l2_normalize_grad_op.set_name(op.op_name() + "_grad");
    L2NormalizeGradOpConf* l2_normalize_grad_op_conf = l2_normalize_grad_op.mutable_l2_normalize_grad_conf();
    l2_normalize_grad_op_conf->set_out(GenLogicalBlobName("out"));
    l2_normalize_grad_op_conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    l2_normalize_grad_op_conf->set_square_x_sum(GenLogicalBlobName("square_x_sum");
    l2_normalize_grad_op_conf->set_in_diff("in_diff");
    l2_normalize_grad_op_conf->set_axis(op.op_conf.l2_normalize_conf().axis());
    l2_normalize_grad_op_conf->set_epsilon(op.op_conf.l2_normalize_conf().epsilon());
    op_confs->push_back(l2_normalize_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(l2_normalize_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("in_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kL2NormalizeConf, &GenerateBackwardOpConf);

}  // namespace oneflow
