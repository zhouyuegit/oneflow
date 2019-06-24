#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_prelu_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf prelu_grad_op;
    prelu_grad_op.set_name(op.op_name() + "_grad");
    PReluGradOpConf* prelu_grad_op_conf = prelu_grad_op.mutable_prelu_grad_conf();
    prelu_grad_op_conf->set_in(GenLogicalBlobName("in"));
    prelu_grad_op_conf->set_alpha(GenLogicalBlobName("alpha"));
    prelu_grad_op_conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    prelu_grad_op_conf->set_in_diff("in_diff");
    prelu_grad_op_conf->set_data_format(op.op_conf().data_format());
    prelu_grad_op_conf->set_channel_shared(op.op_conf().channel_shared());
    op_confs->push_back(prelu_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(prelu_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("in_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kPreluConf, &GenerateBackwardOpConf);

}  // namespace oneflow
