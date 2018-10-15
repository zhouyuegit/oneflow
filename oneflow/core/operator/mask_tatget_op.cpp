#include "oneflow/core/operator/mask_target_op.h"

namespace oneflow {

void MaskTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_mask_target_conf());
  // Enroll input
  EnrollInputBn("sample_rois", false);
  EnrollInputBn("sample_labels", false);
  EnrollInputBn("seg_polys", false);
  EnrollInputBn("seg_cls", false);
  // Enroll output
  EnrollOutputBn("mask_rois", false);
  EnrollOutputBn("masks", false);
  // Enroll data tmp
  // EnrollDataTmpBn("boxes_index");
  // EnrollDataTmpBn("max_overlaps");
  // EnrollDataTmpBn("max_overlaps_gt_boxes_index");
}

const PbMessage& MaskTargetOp::GetCustomizedConf() const { return op_conf().mask_target_conf(); }

void MaskTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  // TODO: Check conf
  // input: sample_rois (n, r, 4) T
  const BlobDesc* sample_rois_blob_desc = GetBlobDesc4BnInOp("sample_rois");
  // input: sample_labels (n * r) int32_t
  const BlobDesc* sample_labels_blob_desc = GetBlobDesc4BnInOp("sample_labels");
  // input: seg_polys (n,g,b) float
  const BlobDesc* seg_polys_blob_desc = GetBlobDesc4BnInOp("seg_polys");
  // input: seg_cls (n,g) int32_t
  const BlobDesc* seg_cls_blob_desc = GetBlobDesc4BnInOp("seg_cls");

  int64_t image_num = sample_rois_blob_desc->shape().At(0);
  CHECK_EQ(image_num, seg_polys_blob_desc->shape().At(0));
  CHECK_EQ(image_num, seg_cls_blob_desc->shape().At(0));
  CHECK_EQ(seg_polys_blob_desc->shape().At(1), seg_cls_blob_desc->shape().At(1));
  int64_t roi_num = sample_rois_blob_desc->shape().At(1);
  CHECK_EQ(roi_num * image_num, sample_labels_blob_desc->shape().Count(0));
  int64_t fg_num = conf.num_rois_per_image();
  int64_t class_num = conf.num_classes();
  int64_t max_gt_boxes_num = conf.max_gt_boxes_num();
  int64_t M = conf.resolution();

  DataType data_type = sample_rois_blob_desc->data_type();
  // output: mask_rois (n * fg_num, 5) T
  BlobDesc* mask_rois_blob_desc = GetBlobDesc4BnInOp("mask_rois");
  mask_rois_blob_desc->mut_shape() = Shape({image_num * fg_num, 5});
  mask_rois_blob_desc->set_data_type(data_type);
  mask_rois_blob_desc->set_has_dim0_valid_num_field(true);
  mask_rois_blob_desc->mut_dim0_inner_shape() = Shape({1, image_num * fg_num});
  // output: masks (n * fg_num, class_num , M , M) T
  BlobDesc* masks_blob_desc = GetBlobDesc4BnInOp("masks");
  masks_blob_desc->mut_shape() = Shape({image_num * fg_num, class_num, M, M});
  masks_blob_desc->set_data_type(data_type);
  masks_blob_desc->set_has_dim0_valid_num_field(true);
  masks_blob_desc->mut_dim0_inner_shape() = Shape({1, image_num * fg_num});
}

REGISTER_OP(OperatorConf::kMaskTargetConf, MaskTargetOp);
}  // namespace oneflow