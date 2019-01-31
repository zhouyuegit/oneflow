#include "oneflow/core/kernel/yolo_box_loss_kernel.h"

namespace oneflow {

template<typename T>
void YoloBoxLossKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutputBlobs(ctx, BnInOp2Blob);
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  FOR_RANGE(int32_t, im_i, 0, bbox_blob->shape().At(0)) {
    auto boxes = CalcBoxesAndGtBoxesMaxOverlaps(im_i, BnInOp2Blob);
    CalcSamplesAndBboxLoss(im_i, boxes, BnInOp2Blob);
  }
}

template<typename T>
void YoloBoxLossKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_loc_diff_blob = BnInOp2Blob("bbox_loc_diff");
  Blob* bbox_diff_blob = BnInOp2Blob(GenDiffBn("bbox"));
  bbox_diff_blob->CopyDataContentFrom(ctx.device_ctx, bbox_loc_diff_blob);
}

template<typename T>
void YoloBoxLossKernel<T>::ClearOutputBlobs(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* bbox_loc_diff_blob = BnInOp2Blob("bbox_loc_diff");
  Blob* pos_inds_blob = BnInOp2Blob("pos_inds");
  Blob* pos_cls_label_blob = BnInOp2Blob("pos_cls_label");
  Blob* neg_inds_blob = BnInOp2Blob("neg_inds");

  std::memset(bbox_loc_diff_blob->mut_dptr(), 0,
              bbox_loc_diff_blob->shape().elem_cnt() * sizeof(T));
  std::memset(pos_cls_label_blob->mut_dptr(), 0,
              pos_cls_label_blob->shape().elem_cnt() * sizeof(int32_t));
  std::fill(pos_inds_blob->mut_dptr<int32_t>(),
            pos_inds_blob->mut_dptr<int32_t>() + pos_inds_blob->shape().elem_cnt(), -1);
  std::fill(neg_inds_blob->mut_dptr<int32_t>(),
            neg_inds_blob->mut_dptr<int32_t>() + neg_inds_blob->shape().elem_cnt(), -1);
}

template<typename T>
typename YoloBoxLossKernel<T>::BoxesWithMaxOverlapSlice
YoloBoxLossKernel<T>::CalcBoxesAndGtBoxesMaxOverlaps(
    int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const YoloBoxLossOpConf& conf = op_conf().yolo_box_loss_conf();
  // Col gt boxes
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const size_t col_num = gt_boxes_blob->dim1_valid_num(im_index);
  const BBox* gt_boxes = BBox::Cast(gt_boxes_blob->dptr<T>(im_index));
  // Row boxes
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const size_t row_num = bbox_blob->shape().At(1);
  BoxesWithMaxOverlapSlice boxes(
      BoxesSlice(IndexSequence(row_num, BnInOp2Blob("bbox_inds")->mut_dptr<int32_t>(), true),
                 bbox_blob->dptr<T>(im_index)),
      BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_gt_indices")->mut_dptr<int32_t>(), true);

  FOR_RANGE(size_t, i, 0, row_num) {
    FOR_RANGE(size_t, j, 0, col_num) {
      const BBox* gt_bbox = gt_boxes + j;
      if (gt_bbox->Area() <= 0) { continue; }
      std::array<T, 4> tmp_pred_buf;
      BBox* pred_bbox = BBox::Cast(tmp_pred_buf.data());
      pred_bbox->set_xywh(boxes.GetBBox(i)->center_x(), boxes.GetBBox(i)->center_y(),
                          boxes.GetBBox(i)->width(), boxes.GetBBox(i)->height());
      BboxCoordinateTransform(boxes.GetIndex(i), pred_bbox);
      const float overlap = pred_bbox->InterOverUnion(gt_bbox);
      int32_t max_overlap_gt_index = -2;                                   // donot care
      if (overlap <= conf.ignore_thresh()) { max_overlap_gt_index = -1; }  // negative
      if (overlap > conf.truth_thresh()) { max_overlap_gt_index = j; }     // postive
      boxes.TryUpdateMaxOverlap(boxes.GetIndex(i), max_overlap_gt_index, overlap);
    }
  }
  std::map<int32_t, int32_t> bias_mask;
  FOR_RANGE(size_t, i, 0, conf.nbox()) { bias_mask[conf.mask(i)] = i; }
  FOR_RANGE(size_t, k, 0, col_num) {
    const BBox* gt_bbox = gt_boxes + k;
    const int32_t fm_i = static_cast<int32_t>(std::floor(gt_bbox->center_x() * conf.layer_width()));
    const int32_t fm_j =
        static_cast<int32_t>(std::floor(gt_bbox->center_y() * conf.layer_height()));
    std::array<T, 4> tmp_gt_buf;
    BBox* tmp_gt_bbox = BBox::Cast(tmp_gt_buf.data());
    tmp_gt_bbox->set_xywh(0, 0, gt_bbox->width(), gt_bbox->height());
    float max_iou = 0.0f;
    int32_t max_iou_pred_index = -1;
    FOR_RANGE(size_t, ibox, 0, conf.total_box()) {
      std::array<T, 4> tmp_pred_buf;
      BBox* pred_box = BBox::Cast(tmp_pred_buf.data());
      pred_box->set_xywh(
          0, 0, static_cast<T>(conf.biases(2 * ibox)) / static_cast<T>(conf.image_width()),
          static_cast<T>(conf.biases(2 * ibox + 1)) / static_cast<T>(conf.image_height()));
      const float iou = pred_box->InterOverUnion(tmp_gt_bbox);
      if (iou > max_iou) {
        max_iou = iou;
        max_iou_pred_index = ibox;
      }
    }
    if (bias_mask.find(max_iou_pred_index) != bias_mask.end()) {
      const int32_t box_index = fm_j * conf.layer_width() * conf.nbox() + fm_i * conf.nbox()
                                + bias_mask[max_iou_pred_index];
      boxes.set_max_overlap_with_index(box_index, k);
    }
  }
  return boxes;
}

template<typename T>
void YoloBoxLossKernel<T>::CalcSamplesAndBboxLoss(
    const int64_t im_index, BoxesWithMaxOverlapSlice& boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* pos_inds_blob = BnInOp2Blob("pos_inds");
  Blob* neg_inds_blob = BnInOp2Blob("neg_inds");
  IndexSequence pos_sample(BnInOp2Blob("bbox")->shape().At(1),
                           pos_inds_blob->mut_dptr<int32_t>(im_index));
  pos_sample.Assign(boxes);
  pos_sample.Filter([&](int32_t index) { return boxes.max_overlap_with_index(index) < 0; });
  pos_inds_blob->set_dim1_valid_num(im_index, pos_sample.size());
  IndexSequence neg_sample(BnInOp2Blob("bbox")->shape().At(1),
                           neg_inds_blob->mut_dptr<int32_t>(im_index));
  neg_sample.Assign(boxes);
  neg_sample.Filter([&](int32_t index) { return boxes.max_overlap_with_index(index) != -1; });
  neg_inds_blob->set_dim1_valid_num(im_index, neg_sample.size());
  boxes.Truncate(0);
  boxes.Concat(pos_sample);
  CalcBboxLoss(im_index, boxes, BnInOp2Blob);
}

template<typename T>
void YoloBoxLossKernel<T>::CalcBboxLoss(
    const int64_t im_index, const BoxesWithMaxOverlapSlice& boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const BBox* gt_boxes = BBox::Cast(BnInOp2Blob("gt_boxes")->dptr<T>(im_index));
  const int32_t* gt_labels_ptr = BnInOp2Blob("gt_labels")->dptr<int32_t>(im_index);
  int32_t* labels_ptr = BnInOp2Blob("pos_cls_label")->mut_dptr<int32_t>(im_index);
  T* bbox_loc_diff_ptr = BnInOp2Blob("bbox_loc_diff")->mut_dptr<T>(im_index);
  FOR_RANGE(size_t, i, 0, boxes.size()) {
    int32_t index = boxes.GetIndex(i);
    int32_t gt_index = boxes.max_overlap_with_index(index);
    labels_ptr[index] = gt_labels_ptr[gt_index];
    const auto* bbox = boxes.bbox(index);
    const float scale = 2 - gt_boxes[gt_index].width() * gt_boxes[gt_index].height();
    std::array<T, 4> tmp_truth_buf;
    BBox* truth_box = BBox::Cast(tmp_truth_buf.data());
    truth_box->set_xywh(gt_boxes[gt_index].center_x(), gt_boxes[gt_index].center_y(),
                        gt_boxes[gt_index].width(), gt_boxes[gt_index].height());
    BboxCoordinateTransformInverse(index, truth_box);
    bbox_loc_diff_ptr[index * 4 + 0] = scale * (truth_box->center_x() - bbox->center_x());
    bbox_loc_diff_ptr[index * 4 + 1] = scale * (truth_box->center_y() - bbox->center_y());
    bbox_loc_diff_ptr[index * 4 + 2] = scale * (truth_box->width() - bbox->width());
    bbox_loc_diff_ptr[index * 4 + 3] = scale * (truth_box->height() - bbox->height());
  }
}

template<typename T>
void YoloBoxLossKernel<T>::BboxCoordinateTransform(const int32_t box_index, BBox* pred_box) const {
  const YoloBoxLossOpConf& conf = op_conf().yolo_box_loss_conf();
  const int32_t iw = (box_index / conf.nbox()) % conf.layer_width();
  const int32_t ih = (box_index / conf.nbox()) / conf.layer_width();
  const int32_t ibox = conf.mask(box_index % conf.nbox());
  T box_x = (pred_box->center_x() + iw) / static_cast<T>(conf.layer_width());
  T box_y = (pred_box->center_y() + ih) / static_cast<T>(conf.layer_height());
  T box_w =
      std::exp(pred_box->width()) * conf.biases(2 * ibox) / static_cast<T>(conf.image_width());
  T box_h = std::exp(pred_box->height()) * conf.biases(2 * ibox + 1)
            / static_cast<T>(conf.image_height());
  pred_box->set_xywh(box_x, box_y, box_w, box_h);
}

template<typename T>
void YoloBoxLossKernel<T>::BboxCoordinateTransformInverse(const int32_t box_index,
                                                          BBox* truth_box) const {
  const YoloBoxLossOpConf& conf = op_conf().yolo_box_loss_conf();
  const int32_t iw = (box_index / conf.nbox()) % conf.layer_width();
  const int32_t ih = (box_index / conf.nbox()) / conf.layer_width();
  const int32_t ibox = conf.mask(box_index % conf.nbox());
  float box_x = truth_box->center_x() * conf.layer_width() - iw;
  float box_y = truth_box->center_y() * conf.layer_height() - ih;
  float box_w =
      std::log(truth_box->width() * conf.image_width() / static_cast<T>(conf.biases(2 * ibox)));
  float box_h = std::log(truth_box->height() * conf.image_height()
                         / static_cast<T>(conf.biases(2 * ibox + 1)));
  truth_box->set_xywh(box_x, box_y, box_w, box_h);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloBoxLossConf, YoloBoxLossKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
