#include "oneflow/core/operator/yolo_prob_loss_op.h"

namespace oneflow {

void YoloProbLossOp::InitFromOpConf() {
  CHECK(op_conf().has_yolo_prob_loss_conf());
  // Enroll input
  EnrollInputBn("prob_logistic");
  EnrollInputBn("pos_cls_label", false);
  EnrollInputBn("pos_inds", false);
  EnrollInputBn("neg_inds", false);

  // Enroll output
  EnrollOutputBn("prob_loss", true);

  // data tmp
  EnrollDataTmpBn("label_tmp");
  EnrollDataTmpBn("prob_diff_tmp");
}

const PbMessage& YoloProbLossOp::GetCustomizedConf() const {
  return op_conf().yolo_prob_loss_conf();
}

void YoloProbLossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  // input: prob_logistic : (n, r, 81)  r = h*w*3
  const BlobDesc* prob_logistic_blob_desc = GetBlobDesc4BnInOp("prob_logistic");
  // input: pos_cls_label (n, r)
  const BlobDesc* pos_cls_label_blob_desc = GetBlobDesc4BnInOp("pos_cls_label");
  // input: pos_inds (n, r) int32_t
  const BlobDesc* pos_inds_blob_desc = GetBlobDesc4BnInOp("pos_inds");
  // input: neg_inds (n, r) int32_t
  const BlobDesc* neg_inds_blob_desc = GetBlobDesc4BnInOp("neg_inds");

  const int64_t num_images = prob_logistic_blob_desc->shape().At(0);
  CHECK_EQ(num_images, pos_cls_label_blob_desc->shape().At(0));
  CHECK_EQ(num_images, pos_inds_blob_desc->shape().At(0));
  CHECK_EQ(num_images, neg_inds_blob_desc->shape().At(0));
  const int64_t num_boxes = prob_logistic_blob_desc->shape().At(1);
  const int64_t num_probs = 1 + op_conf().yolo_prob_loss_conf().num_classes();
  CHECK_EQ(num_boxes, pos_cls_label_blob_desc->shape().At(1));
  CHECK_EQ(num_boxes, pos_inds_blob_desc->shape().At(1));
  CHECK_EQ(num_boxes, neg_inds_blob_desc->shape().At(1));
  CHECK_EQ(num_probs, prob_logistic_blob_desc->shape().At(2));

  // output: prob_loss (n, r)
  BlobDesc* prob_loss_blob_desc = GetBlobDesc4BnInOp("prob_loss");
  prob_loss_blob_desc->mut_shape() = Shape({num_images, num_boxes});
  prob_loss_blob_desc->set_data_type(prob_logistic_blob_desc->data_type());

  // tmp: label_tmp (81) int32_t
  BlobDesc* label_tmp_blob_desc = GetBlobDesc4BnInOp("label_tmp");
  label_tmp_blob_desc->mut_shape() = Shape({num_probs});
  label_tmp_blob_desc->set_data_type(DataType::kInt32);

  // tmp: prob_diff_tmp (n, r, 81)
  BlobDesc* prob_diff_tmp_blob_desc = GetBlobDesc4BnInOp("prob_diff_tmp");
  prob_diff_tmp_blob_desc->mut_shape() = Shape({num_images, num_boxes, num_probs});
  prob_diff_tmp_blob_desc->set_data_type(prob_logistic_blob_desc->data_type());
}

// REGISTER_OP(OperatorConf::kYoloProbLossConf, YoloProbLossOp);
REGISTER_CPU_OP(OperatorConf::kYoloProbLossConf, YoloProbLossOp);
}  // namespace oneflow
