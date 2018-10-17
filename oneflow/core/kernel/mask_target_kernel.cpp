#include "oneflow/core/kernel/mask_target_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"
extern "C" {
#include <maskApi.h>
}

namespace oneflow {

namespace {

auto GetValidDim0 = [](const Blob* blob, int32_t no) -> int32_t {
  if (blob->dim1_valid_num() != nullptr) { return blob->dim0_valid_num(no); }
  return blob->shape().At(1);
};

auto GetValidDim1 = [](const Blob* blob, int32_t no) -> int32_t {
  if (blob->dim1_valid_num() != nullptr) { return blob->dim1_valid_num(no); }
  return blob->shape().At(1);
};

auto GetValidDim2 = [](const Blob* blob, int32_t no0, int32_t no1) -> int32_t {
  if (blob->dim2_valid_num() != nullptr) { return blob->dim2_valid_num(no0, no1); }
  return blob->shape().At(2);
};

auto SetValidDim0 = [](Blob* blob, int32_t no, int32_t val) -> void {
  blob->set_dim0_valid_num(no, val);
};

}  // namespace

template<typename T>
void MaskTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GetMaskBoxes(BnInOp2Blob);
  int32_t valid_rois_num = GetValidDim0(BnInOp2Blob("in_rois"), 0);
  FOR_RANGE(int32_t, i, 0, valid_rois_num) {
    const Blob* in_rois_blob = BnInOp2Blob("in_rois");
    const Blob* in_labels_blob = BnInOp2Blob("in_labels");
    Blob* mask_rois_blob = BnInOp2Blob("mask_rois");
    Blob* masks_blob = BnInOp2Blob("masks");
    int fg_num = 0;
    if (in_labels_blob->dptr<int32_t>(i) > 0) {  // if roi is fg
      T cur_image = in_rois_blob->dptr<T>(i)[0];
      int32_t max_overlap_gt_index = GetMaxOverlapMaskBoxIndex(cur_image, i, BnInOp2Blob);
      Polys2MaskWrtBox(cur_image, max_overlap_gt_index, i, BnInOp2Blob);
      // output mask_rois
      T* mask_roi = mask_rois_blob->mut_dptr<T>(fg_num);
      mask_roi[0] = in_rois_blob->dptr<T>(i)[0];
      mask_roi[1] = in_rois_blob->dptr<T>(i)[1];
      mask_roi[2] = in_rois_blob->dptr<T>(i)[2];
      mask_roi[3] = in_rois_blob->dptr<T>(i)[3];
      mask_roi[4] = in_rois_blob->dptr<T>(i)[4];
      fg_num++;
    }
    SetValidDim0(mask_rois_blob, 0, fg_num);
    SetValidDim0(masks_blob, 0, fg_num);
  }
}

template<typename T>
void MaskTargetKernel<T>::GetMaskBoxes(
    const std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* seg_polys_blob = BnInOp2Blob("gt_segm_polygon_lists");
  Blob* mask_boxes_blob = BnInOp2Blob("mask_boxes");
  FOR_RANGE(int64_t, im_index, 0, BnInOp2Blob("in_rois")->shape().At(0)) {
    int32_t valid_polys_num = GetValidDim1(seg_polys_blob, im_index);
    FOR_RANGE(int32_t, gt_index, 0, valid_polys_num) {
      int32_t valid_polys_length = GetValidDim2(seg_polys_blob, im_index, gt_index);
      PolygonList polys;
      polys.ParseFromArray(seg_polys_blob->dptr<char>(im_index, gt_index), valid_polys_length);
      float x0 = 0;
      float x1 = 0;
      float y0 = 0;
      float y1 = 0;
      // gt might contain several polys
      FOR_RANGE(int32_t, poly_index, 0, polys.polygons_size()) {
        FOR_RANGE(int32_t, k, 0, polys.polygons(poly_index).value_size()) {
          if (k % 2 == 0) {
            x0 = std::min(x0, polys.polygons(poly_index).value(k));
            x1 = std::max(x1, polys.polygons(poly_index).value(k));
          } else {
            y0 = std::min(x0, polys.polygons(poly_index).value(k));
            y1 = std::max(x1, polys.polygons(poly_index).value(k));
          }
        }
      }
      float* mask_box = mask_boxes_blob->mut_dptr<float>(im_index, gt_index);
      mask_box[0] = x0;
      mask_box[1] = x1;
      mask_box[2] = y0;
      mask_box[3] = y1;
    }
  }
}

template<typename T>
int32_t MaskTargetKernel<T>::GetMaxOverlapMaskBoxIndex(
    T im_index, int32_t roi_index,
    const std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* mask_boxes_blob = BnInOp2Blob("mask_boxes");
  const Blob* in_rois_blob = BnInOp2Blob("in_rois");
  const Blob* seg_polys_blob = BnInOp2Blob("gt_segm_polygon_lists");
  const T* in_roi_ptr = in_rois_blob->dptr<T>(roi_index);
  const BBox2<T>* roi_box = BBox2<T>::Cast(in_roi_ptr);

  int32_t valid_polys_num = GetValidDim1(seg_polys_blob, im_index);
  float max_overlap = 0;
  int32_t max_overlap_gt_ind = 0;
  FOR_RANGE(int32_t, gt_index, 0, valid_polys_num) {
    const float* mask_box_ptr = mask_boxes_blob->dptr<float>(im_index, gt_index);
    const BBox<float>* mask_box = BBox<float>::Cast(mask_box_ptr);
    float iou = roi_box->InterOverUnion(mask_box);  // bbox2 add iou func
    if (iou > max_overlap) {
      max_overlap = iou;
      max_overlap_gt_ind = gt_index;
    }
  }
  return max_overlap_gt_ind;
}

template<typename T>
void MaskTargetKernel<T>::Polys2MaskWrtBox(
    T im_index, int32_t gt_index, int32_t roi_index,
    const std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* fg_rois_blob = BnInOp2Blob("in_rois");
  const Blob* seg_polys_blob = BnInOp2Blob("gt_segm_polygon_lists");
  const Blob* seg_cls_blob = BnInOp2Blob("gt_segm_labels");
  Blob* masks_blob = BnInOp2Blob("masks");

  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  int32_t M = conf.resolution();

  int32_t valid_polys_length = GetValidDim2(seg_polys_blob, im_index, gt_index);
  PolygonList polys;
  polys.ParseFromArray(seg_polys_blob->dptr<char>(im_index, gt_index), valid_polys_length);
  const T* fg_roi_ptr = fg_rois_blob->dptr<T>(roi_index);
  const BBox2<T>* fg_box = BBox2<T>::Cast(fg_roi_ptr);

  T w = fg_box->width();
  T h = fg_box->height();
  std::vector<byte> mask(M * M);
  FOR_RANGE(int32_t, poly_index, 0, polys.polygons_size()) {
    double* poly = new double[polys.polygons(poly_index).value_size()];
    FOR_RANGE(int32_t, k, 0, polys.polygons(poly_index).value_size()) {
      if (k % 2 == 0) {
        poly[k] = (polys.polygons(poly_index).value(k) - fg_box->x1()) * M / w;
      } else {
        poly[k] = (polys.polygons(poly_index).value(k) - fg_box->y1()) * M / h;
      }
    }

    std::vector<byte> mask_k(M * M);
    RLE rle;
    rleFrPoly(&rle, poly, polys.polygons(poly_index).value_size(), M, M);
    delete[] poly;
    rleDecode(&rle, mask_k.data(), 1);
    FOR_RANGE(int32_t, j, 0, M * M) { mask[j] |= mask_k[j]; }
  }
  // output mask
  const int32_t cls = seg_cls_blob->dptr<int32_t>(im_index, gt_index)[0];
  T* mask_ptr = masks_blob->mut_dptr<T>(roi_index, cls);
  FOR_RANGE(int32_t, row, 0, M) {
    FOR_RANGE(int32_t, col, 0, M) {
      mask_ptr[row * M + col] = mask[col * M + row];  // transpose
    }
  }
}

template<typename T>
void MaskTargetKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaskTargetConf, MaskTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
