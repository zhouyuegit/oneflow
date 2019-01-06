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
  TODO();
}

template<typename T>
void YoloBoxLossKernel<T>::ClearOutputBlobs(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* box_loss_blob = BnInOp2Blob("box_loss");
  Blob* pos_inds_blob = BnInOp2Blob("pos_inds");
  Blob* pos_cls_label_blob = BnInOp2Blob("pos_cls_label");
  Blob* neg_inds_blob = BnInOp2Blob("neg_inds");
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* max_overlaps_gt_indices_blob = BnInOp2Blob("max_overlaps_gt_indices");

  std::memset(box_loss_blob->mut_dptr(), 0, box_loss_blob->shape().elem_cnt() * sizeof(T));
  std::memset(pos_inds_blob->mut_dptr(), -1, pos_inds_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(pos_cls_label_blob->mut_dptr(), 0,
              pos_cls_label_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(neg_inds_blob->mut_dptr(), -1, neg_inds_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(max_overlaps_blob->mut_dptr(), 0,
              max_overlaps_blob->shape().elem_cnt() * sizeof(float));
  std::memset(max_overlaps_gt_indices_blob->mut_dptr(), 0,
              max_overlaps_gt_indices_blob->shape().elem_cnt() * sizeof(int32_t));
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
  Blob* bbox_inds_blob = BnInOp2Blob("bbox_inds");
  const size_t row_num = bbox_blob->shape().At(1);
  BoxesWithMaxOverlapSlice boxes(
      BoxesSlice(IndexSequence(row_num, bbox_inds_blob->mut_dptr<int32_t>(im_index), true),
                 bbox_blob->dptr<T>(im_index)),
      BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_gt_indices")->mut_dptr<int32_t>(), true);

  FOR_RANGE(size_t, i, 0, row_num) {
    FOR_RANGE(size_t, j, 0, col_num) {
      const BBox* gt_bbox = gt_boxes + j;
      if (gt_bbox->Area() <= 0) { continue; }
      PredBoxTransform(boxes.GetIndex(i), boxes.GetBBox(i));
      const float overlap = boxes.GetBBox(i)->InterOverUnion(gt_bbox);
      int32_t max_overlap_gt_index = -2;                                   // donot care
      if (overlap <= conf.ignore_thresh()) { max_overlap_gt_index = -1; }  // negative
      if (overlap > conf.truth_thresh()) { max_overlap_gt_index = j; }     // postive
      boxes.TryUpdateMaxOverlap(boxes.GetIndex(i), max_overlap_gt_index, overlap);
    }
  }
  std::set<int32_t> bias_mask;
  FOR_RANGE(size_t, i, 0, conf.nbox()) { bias_mask.insert(conf.mask(i)); }
  FOR_RANGE(size_t, k, 0, col_num) {
    const BBox* gt_bbox = gt_boxes + k;
    const int fm_i = gt_bbox->center_x() * conf.layer_width();   // float->int
    const int fm_j = gt_bbox->center_y() * conf.layer_height();  // float->int
    std::vector<T> tmp_mem(4);
    BBox* tmp_gt_bbox = BBox::Cast(tmp_mem.data());
    tmp_gt_bbox->set_xywh(0, 0, gt_bbox->width(), gt_bbox->height());
    float max_iou = 0.0f;
    int32_t max_iou_pred_index = -1;
    FOR_RANGE(size_t, ibox, 0, conf.total_box()) {
      std::vector<T> tmp_mem(4);
      BBox* pred_box = BBox::Cast(tmp_mem.data());
      pred_box->set_xywh(0, 0, conf.biases(2 * ibox), conf.biases(2 * ibox + 1));
      const float iou = pred_box->InterOverUnion(tmp_gt_bbox);
      if (iou > max_iou) {
        max_iou = iou;
        max_iou_pred_index = ibox;
      }
    }
    if (bias_mask.find(max_iou_pred_index) != bias_mask.end()) {
      const int32_t box_index = fm_i * conf.layer_width() + fm_j;
      boxes.set_max_overlap_with_index(box_index, k);
    }
  }
  return boxes;
}

template<typename T>
void YoloBoxLossKernel<T>::CalcSamplesAndBboxLoss(
    const int64_t im_index, BoxesWithMaxOverlapSlice& boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  IndexSequence pos_sample(BnInOp2Blob("bbox")->shape().At(1),
                           BnInOp2Blob("pos_inds")->mut_dptr<int32_t>(im_index));
  pos_sample.Assign(boxes);
  pos_sample.Filter([&](int32_t index) { return boxes.max_overlap_with_index(index) < 0; });
  IndexSequence neg_sample(BnInOp2Blob("bbox")->shape().At(1),
                           BnInOp2Blob("neg_inds")->mut_dptr<int32_t>());
  neg_sample.Assign(boxes);
  neg_sample.Filter([&](int32_t index) { return boxes.max_overlap_with_index(index) != -1; });
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
  T* box_loss_ptr = BnInOp2Blob("box_loss")->mut_dptr<T>(im_index);
  FOR_RANGE(size_t, i, 0, boxes.size()) {
    int32_t index = boxes.GetIndex(i);
    int32_t gt_index = boxes.max_overlap_with_index(index);
    labels_ptr[index] = gt_labels_ptr[gt_index];
    const auto* bbox = boxes.bbox(index);
    const float scale = 2 - gt_boxes[gt_index].width() * gt_boxes[gt_index].height();
    std::vector<T> tmp_mem(4);
    BBox* truth_box = BBox::Cast(tmp_mem.data());
    truth_box->set_xywh(gt_boxes[gt_index].center_x(), gt_boxes[gt_index].center_y(),
                        gt_boxes[gt_index].width(), gt_boxes[gt_index].height());
    TruthBoxTransformInverse(index, truth_box);
    box_loss_ptr[index * 4 + 0] = scale * (truth_box->center_x() - bbox->center_x());
    box_loss_ptr[index * 4 + 1] = scale * (truth_box->center_x() - bbox->center_x());
    box_loss_ptr[index * 4 + 2] = scale * (truth_box->width() - bbox->width());
    box_loss_ptr[index * 4 + 3] = scale * (truth_box->height() - bbox->height());
  }
}

template<typename T>
void YoloBoxLossKernel<T>::PredBoxTransform(const int32_t box_index, BBox* pred_box) const {
  const YoloBoxLossOpConf& conf = op_conf().yolo_box_loss_conf();
  const int32_t iw = (box_index / conf.nbox()) % conf.layer_width();
  const int32_t ih = (box_index / conf.nbox()) / conf.layer_width();
  const int32_t ibox = conf.mask(box_index % conf.nbox());
  float box_x = (pred_box->center_x() + iw) / conf.layer_width();
  float box_y = (pred_box->center_y() + ih) / conf.layer_height();
  float box_w = std::exp(pred_box->width()) * conf.biases(2 * ibox) / conf.image_width();
  float box_h = std::exp(pred_box->height()) * conf.biases(2 * ibox + 1) / conf.image_height();
  pred_box->set_xywh(box_x, box_y, box_w, box_h);
}

template<typename T>
void YoloBoxLossKernel<T>::TruthBoxTransformInverse(const int32_t box_index,
                                                    BBox* truth_box) const {
  const YoloBoxLossOpConf& conf = op_conf().yolo_box_loss_conf();
  const int32_t iw = (box_index / conf.nbox()) % conf.layer_width();
  const int32_t ih = (box_index / conf.nbox()) / conf.layer_width();
  const int32_t ibox = conf.mask(box_index % conf.nbox());
  float box_x = truth_box->center_x() * conf.layer_width() - iw;
  float box_y = truth_box->center_y() * conf.layer_height() - ih;
  float box_w = std::log(truth_box->width() * conf.image_width() / conf.biases(2 * ibox));
  float box_h = std::log(truth_box->height() * conf.image_height() / conf.biases(2 * ibox + 1));
  truth_box->set_xywh(box_x, box_y, box_w, box_h);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloBoxLossConf, YoloBoxLossKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
