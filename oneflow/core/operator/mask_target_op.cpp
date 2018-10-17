#include "oneflow/core/operator/mask_target_op.h"

namespace oneflow {

void MaskTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_mask_target_conf());
  // Enroll input
  EnrollInputBn("in_rois", false);
  EnrollInputBn("in_labels", false);
  EnrollInputBn("gt_segm_polygon_lists", false);
  EnrollInputBn("gt_segm_labels", false);
  // Enroll output
  EnrollOutputBn("mask_rois", false);
  EnrollOutputBn("masks", false);
  // Enroll data tmp
  EnrollDataTmpBn("mask_boxes");
}

const PbMessage& MaskTargetOp::GetCustomizedConf() const { return op_conf().mask_target_conf(); }

void MaskTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  // TODO: Check conf
  // input: in_rois (R, 5) T
  const BlobDesc* in_rois_blob_desc = GetBlobDesc4BnInOp("in_rois");
  // input: in_labels (R) int32_t
  const BlobDesc* in_labels_blob_desc = GetBlobDesc4BnInOp("in_labels");
  // input: gt_segm_polygon_lists (N,G,B) float
  const BlobDesc* seg_polys_blob_desc = GetBlobDesc4BnInOp("gt_segm_polygon_lists");
  // input: gt_segm_labels (N,G) int32_t
  const BlobDesc* seg_cls_blob_desc = GetBlobDesc4BnInOp("gt_segm_labels");

  CHECK_EQ(seg_polys_blob_desc->shape().At(1), seg_cls_blob_desc->shape().At(1));
  int64_t R = in_rois_blob_desc->shape().At(0);
  int64_t N = seg_polys_blob_desc->shape().At(0);
  int64_t G = seg_polys_blob_desc->shape().At(1);
  int64_t M = conf.resolution();
  int64_t class_num = conf.num_classes();

  DataType data_type = in_rois_blob_desc->data_type();
  // output: mask_rois (fgR, 5) T
  BlobDesc* mask_rois_blob_desc = GetBlobDesc4BnInOp("mask_rois");
  mask_rois_blob_desc->mut_shape() = Shape({R, 5});
  mask_rois_blob_desc->set_data_type(data_type);
  mask_rois_blob_desc->set_has_dim0_valid_num_field(true);
  mask_rois_blob_desc->mut_dim0_inner_shape() = Shape({1, R});
  // output: masks (n * fg_num, class_num , M , M) T
  BlobDesc* masks_blob_desc = GetBlobDesc4BnInOp("masks");
  masks_blob_desc->mut_shape() = Shape({R, class_num, M, M});
  masks_blob_desc->set_data_type(data_type);
  masks_blob_desc->set_has_dim0_valid_num_field(true);
  masks_blob_desc->mut_dim0_inner_shape() = Shape({1, R});
  // data tmp: mask_boxes (N,G,4) float
  BlobDesc* mask_boxes_blob_desc = GetBlobDesc4BnInOp("mask_boxes");
  mask_boxes_blob_desc->mut_shape() = Shape({N, G, 4});
  mask_boxes_blob_desc->set_data_type(DataType::kFloat);
}

REGISTER_OP(OperatorConf::kMaskTargetConf, MaskTargetOp);
}  // namespace oneflow