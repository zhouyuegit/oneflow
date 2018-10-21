#include "oneflow/core/operator/bbox_vote_op.h"

namespace oneflow {

void BboxVoteOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_transform_conf());
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_score", false);
  EnrollInputBn("bbox_label", false);
  EnrollOutputBn("out_bbox", false);
  EnrollOutputBn("out_bbox_score", false);
  EnrollOutputBn("out_bbox_label", false);
}

const PbMessage& BboxVoteOp::GetCustomizedConf() const { return op_conf().bbox_vote_conf(); }

void BboxVoteOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  *GetBlobDesc4BnInOp("out_bbox") = *bbox_blob_desc;

  const BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  *GetBlobDesc4BnInOp("out_bbox_score") = *bbox_score_blob_desc;
  *GetBlobDesc4BnInOp("out_bbox_label") = *bbox_score_blob_desc;
}

REGISTER_OP(OperatorConf::kBboxVoteConf, BboxVoteOp);

}  // namespace oneflow