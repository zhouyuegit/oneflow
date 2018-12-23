#include "oneflow/core/operator/yolo_box_op.h"

namespace oneflow {

void YoloBoxOp::InitFromOpConf() {
  CHECK(op_conf().has_yolo_box_conf());
  EnrollInputBn("bbox");
  EnrollInputBn("probs");
  EnrollOutputBn("out_bbox");
  EnrollOutputBn("out_probs");
  EnrollDataTmpBn("probs_index");
}

const PbMessage& YoloBoxOp::GetCustomizedConf() const { return op_conf().yolo_box_conf(); }

void YoloBoxOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // bbox : (n, h*w*3, 4)
  // probs : (n, h*w*3, 81)
  // out_bbox : (n, h*w*3, 4)
  // out_probs : (n, h*w*3, 81)
  // probs_index : (h*w*3)
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  const int32_t num_boxes = bbox_blob_desc->shape().At(1);
  *GetBlobDesc4BnInOp("out_bbox") = *bbox_blob_desc;
  *GetBlobDesc4BnInOp("out_probs") = *GetBlobDesc4BnInOp("probs");

  // data_tmp: probs_index (h*w*3) float
  BlobDesc* probs_index_blob_desc = GetBlobDesc4BnInOp("probs_index");
  probs_index_blob_desc->set_data_type(DataType::kInt32);
  probs_index_blob_desc->mut_shape() = Shape({num_boxes});
}

REGISTER_OP(OperatorConf::kYoloBoxConf, YoloBoxOp);

}  // namespace oneflow
