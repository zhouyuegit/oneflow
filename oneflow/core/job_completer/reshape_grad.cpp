#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_reshape_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reverse_reshape_op;
    reverse_reshape_op.set_name(op.op_name() + "_grad");
    const BlobDesc& in_logical_blob_desc = LogicalBlobDesc4BnInOp("in");
    std::string in_diff_blob_name;
    if (in_logical_blob_desc.has_dim0_valid_num_field()) {
      ReshapeLikeOpConf* reshape_like_op_conf = reverse_reshape_op.mutable_reshape_like_conf();
      reshape_like_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reshape_like_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
      reshape_like_op_conf->set_y("y");
      in_diff_blob_name = "y";
    } else {
      ReshapeOpConf* reshape_conf = reverse_reshape_op.mutable_reshape_conf();
      reshape_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reshape_conf->set_out("out");
      in_logical_blob_desc.shape().ToProto(reshape_conf->mutable_shape());
      in_diff_blob_name = "out";
    }
    op_confs->push_back(reverse_reshape_op);
    DiffLbi4BnInOp("in")->set_op_name(reverse_reshape_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(in_diff_blob_name);
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReshapeConf, &GenerateBackwardOpConf);

}  // namespace oneflow
