#include "oneflow/core/operator/bbox_nms_op.h"

namespace oneflow {

void BboxNmsOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_nms_conf());
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_prob", false);
  EnrollDataTmpBn("bbox_score");
  EnrollOutputBn("out_bbox", false);
  EnrollOutputBn("out_bbox_score", false);
  EnrollOutputBn("out_bbox_label", false);
}

const PbMessage& BboxNmsOp::GetCustomizedConf() const { return op_conf().bbox_nms_conf(); }

void BboxNmsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  *GetBlobDesc4BnInOp("out_bbox") = *bbox_blob_desc;

  const BlobDesc* bbox_prob_blob_desc = GetBlobDesc4BnInOp("bbox_prob");
  *GetBlobDesc4BnInOp("bbox_score") = *bbox_prob_blob_desc;
  *GetBlobDesc4BnInOp("out_bbox_score") = *bbox_prob_blob_desc;
  *GetBlobDesc4BnInOp("out_bbox_label") = *bbox_prob_blob_desc;
}

REGISTER_OP(OperatorConf::kBboxNmsConf, BboxNmsOp);

}  // namespace oneflow