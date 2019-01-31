#include "oneflow/core/operator/yolo_box_loss_op.h"

namespace oneflow {

void YoloBoxLossOp::InitFromOpConf() {
  CHECK(op_conf().has_yolo_box_loss_conf());
  // Enroll input
  EnrollInputBn("bbox", true);
  EnrollInputBn("gt_boxes", false);
  EnrollInputBn("gt_labels", false);
  // Enroll output
  EnrollOutputBn("bbox_loc_diff", true);
  EnrollOutputBn("pos_inds", false);
  EnrollOutputBn("pos_cls_label", false);
  EnrollOutputBn("neg_inds", false);
  // data tmp
  EnrollDataTmpBn("bbox_inds");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_gt_indices");
}

const PbMessage& YoloBoxLossOp::GetCustomizedConf() const { return op_conf().yolo_box_loss_conf(); }

void YoloBoxLossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  // input: bbox : (n, r, 4)  r = h*w*3
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  // input: gt_boxes (n, g, 4) T
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // input: gt_labels (n, g) int32_t
  const BlobDesc* gt_labels_blob_desc = GetBlobDesc4BnInOp("gt_labels");
  const int64_t num_images = bbox_blob_desc->shape().At(0);
  CHECK_EQ(num_images, gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(num_images, gt_labels_blob_desc->shape().At(0));
  const int64_t num_boxes = bbox_blob_desc->shape().At(1);
  const int64_t max_num_gt_boxes = gt_boxes_blob_desc->shape().At(1);
  CHECK_EQ(max_num_gt_boxes, gt_labels_blob_desc->shape().At(1));
  CHECK_EQ(bbox_blob_desc->data_type(), gt_boxes_blob_desc->data_type());
  CHECK(gt_boxes_blob_desc->has_dim1_valid_num_field());
  CHECK(gt_labels_blob_desc->has_dim1_valid_num_field());

  // output: bbox_loc_diff (n, r, 4)
  BlobDesc* bbox_loc_diff_blob_desc = GetBlobDesc4BnInOp("bbox_loc_diff");
  bbox_loc_diff_blob_desc->mut_shape() = Shape({num_images, num_boxes, 4});
  bbox_loc_diff_blob_desc->set_data_type(bbox_blob_desc->data_type());
  // output: pos_cls_label (n, r)
  BlobDesc* pos_cls_label_blob_desc = GetBlobDesc4BnInOp("pos_cls_label");
  pos_cls_label_blob_desc->mut_shape() = Shape({num_images, num_boxes});
  pos_cls_label_blob_desc->set_data_type(DataType::kInt32);
  // output: pos_inds (n, r) dynamic
  BlobDesc* pos_inds_blob_desc = GetBlobDesc4BnInOp("pos_inds");
  pos_inds_blob_desc->mut_shape() = Shape({num_images, num_boxes});
  pos_inds_blob_desc->set_data_type(DataType::kInt32);
  pos_inds_blob_desc->set_has_dim1_valid_num_field(true);
  // output: neg_inds (n, r) dynamic
  BlobDesc* neg_inds_blob_desc = GetBlobDesc4BnInOp("neg_inds");
  neg_inds_blob_desc->mut_shape() = Shape({num_images, num_boxes});
  neg_inds_blob_desc->set_data_type(DataType::kInt32);
  neg_inds_blob_desc->set_has_dim1_valid_num_field(true);

  // tmp: bbox_inds (r) int32_t
  BlobDesc* bbox_inds_blob_desc = GetBlobDesc4BnInOp("bbox_inds");
  bbox_inds_blob_desc->mut_shape() = Shape({num_boxes});
  bbox_inds_blob_desc->set_data_type(DataType::kInt32);
  // tmp: max_overlaps (r) float
  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->mut_shape() = Shape({num_boxes});
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  // tmp: max_overlaps_gt_indices (r) int32_t
  BlobDesc* max_overlaps_gt_indices_blob_desc = GetBlobDesc4BnInOp("max_overlaps_gt_indices");
  max_overlaps_gt_indices_blob_desc->mut_shape() = Shape({num_boxes});
  max_overlaps_gt_indices_blob_desc->set_data_type(DataType::kInt32);
}

// REGISTER_OP(OperatorConf::kYoloBoxLossConf, YoloBoxLossOp);
REGISTER_CPU_OP(OperatorConf::kYoloBoxLossConf, YoloBoxLossOp);
}  // namespace oneflow
