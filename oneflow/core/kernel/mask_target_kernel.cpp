#include "oneflow/core/kernel/mask_target_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"
extern C{
  #include <MaskApi.h>
}

namespace oneflow {

namespace{

  auto GetValidDim0 = [](const Blob* blob, int32_t no) -> int32_t {
      if (blob->dim1_valid_num() != nullptr) { return blob->dim0_valid_num(no); }
      return blob->shape().At(1);
  };

  auto GetValidDim1 = [](const Blob* blob, int32_t no) -> int32_t {
      if (blob->dim1_valid_num() != nullptr) { return blob->dim1_valid_num(no); }
      return blob->shape().At(1);
  };

  auto GetValidDim2 = [](const Blob* blob, int32_t no) -> int32_t {
      if (blob->dim2_valid_num() != nullptr) { return blob->dim2_valid_num(no); }
      return blob->shape().At(2);
  };
  
  auto SetValidDim0 = [](const Blob* blob, int32_t no, int32_t val) -> void {
      blob->set_dim0_valid_num(no, val);
  };
}

template<DeviceType device_type, typename T>
void MaskTargetKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(int64_t, i, 0, BnInOp2Blob("sample_rois")->shape().At(0)) {
    auto mask_boxes = GetMaskBoxes(i, BnInOp2Blob);
    auto fg_boxes = GetFgBoxes(i, BnInOp2Blob);
    ComputeFgBoxesAndMaskBoxesOverlaps(mask_boxes, fg_boxes);
    Polys2MaskWrtBox(i, fg_boxes, BnInOp2Blob);
  }
}

template<typename T>
MaskBoxes MaskTargetKernel<T>::GetMaskBoxes(size_t im_index, 
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const{
  Blob* seg_polys_blob = BnInOp2Blob("seg_polys");
  FloatList16 mask_boxes; 
  int32_t valid_polys_num = GetValidDim1(seg_polys_blob, im_index);
  //get one mask_box for each gt
  FOR_RANGE(int32_t, gt_index, 0, valid_polys_num){
    int32_t valid_polys_length = GetValidDim2(seg_polys_blob, im_index, gt_index);
    PolygonList polys;
    polys.ParseFromArray(
      seg_polys_blob->dptr<char>(im_index, gt_index), valid_polys_length);
    float x0 = 0;
    float x1 = 0;
    float y0 = 0;
    float y1 = 0;
    //gt might contain several polys
    FOR_RANGE(int32_t, poly_index, 0, polys.polygons_size()){
      FOR_RANGE(int32_t, k, 0, polys.polygons(poly_index).value_size()){
        if(k % 2 == 0){
          x0 = std::min(x0, polys.polygons(poly_index).value(k));
          x1 = std::max(x1, polys.polygons(poly_index).value(k)); 
        }else{
          y0 = std::min(x0, polys.polygons(poly_index).value(k));
          y1 = std::max(x1, polys.polygons(poly_index).value(k)); 
        }
      }
    }
    mask_boxes.value().add_value(x0);
    mask_boxes.value().add_value(x1);
    mask_boxes.value().add_value(y0);
    mask_boxes.value().add_value(y1); 
  }
  // gen mask boxes
  MaskBoxes mask_boxes(*mask_boxes)
  return mask_boxes; 
}

template<typename T>
BoxesWithMaxOverlap MaskTargetKernel<T>::GetFgBoxes(size_t im_index, 
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const{
  Blob* rois_blob = BnInOp2Blob("sample_rois");
  Blob* labels_blob = BnInOp2Blob("sample_labels");
  Blob* rois_index_blob = BnInOp2Blob("sample_rois_index");
  Blob* mask_rois_blob = BnInOp2Blob("mask_rois");
  auto boxes =
      GenBoxesIndex(rois_index_blob->shape().elem_cnt(), rois_index_blob->mut_dptr<int32_t>(),
                    rois_blob->dptr<T>(im_index), true);
  int32_t valid_rois_num = GetValidDim1(seg_polys, im_index);
  boxes.Truncate(valid_rois_num);
  
  // filter fg
  auto FgFilter = [&](size_t i, int32_t cur_index)->bool{//
    CHECK_EQ(i, cur_index);
    if(labels_blob->dptr<int32_t>(im_index, i) > 0){ return false; }
    return true;
  };
  boxes.Filter(FgFilter);
  // output rois
  int32_t offset = GetValidDim0(mask_rois_blob, 0);//need init? -1 or not?
  FOR_RANGE(int32_t, j, 0, boxes.size()){
    T* mask_rois = mask_rois_blob->mut_dptr<T>();
    const BBox<T>* bbox = boxes.GetBBox(j);
    mask_rois[offset * 5 + 0] = im_index;
    mask_rois[offset * 5 + 1] = bbox->x1();
    mask_rois[offset * 5 + 2] = bbox->y1();
    mask_rois[offset * 5 + 3] = bbox->x2();
    mask_rois[offset * 5 + 4] = bbox->y2();
    offset++;
  }
  SetValidDim0(mask_rois_blob, 0, offset);
  // gen fg_rois with overlap
  BoxesWithMaxOverlap boxes_with_max_overlap(
      boxes, BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_mask_boxes_index")->mut_dptr<int32_t>(), true);

  return boxes_with_max_overlap;
}

template<typename T>
void MaskTargetKernel<T>::ComputeFgBoxesAndMaskBoxesOverlaps(
  const Maskboxes& mask_boxes, BoxesWithMaxOverlap& fg_boxes) const{
  FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
      fg_boxes, mask_boxes, [&](int32_t index, int32_t gt_index, float overlap) {
        fg_boxes.UpdateMaxOverlap(index, gt_index, overlap);
      });
}

template<typename T>
void MaskTargetKernel<T>::Polys2MaskWrtBox(size_t im_index, BoxesWithMaxOverlap& fg_boxes, 
  const std::function<Blob*(const std::string&)>& BnInOp2Blob) const{
  const Blob* seg_polys_blob = BnInOp2Blob("seg_polys");
  const Blob* seg_cls_blob = BnInOp2Blob("seg_cls");
  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  Blob* masks_blob = BnInOp2Blob("masks");
  M = conf.resolution();

  FOR_RANGE(int32_t, i, 0, fg_boxes.size()){
    //fetch max overlap poly and corresponding fg 
    int32_t gt_index = fg_boxes.GetMaxOverlapGtIndex(i);
    int32_t valid_polys_length = GetValidDim2(seg_polys_blob, im_index, gt_index);
    PolygonList polys;
    polys.ParseFromArray(
      seg_polys_blob->dptr<char>(im_index, gt_index), valid_polys_length);
    BBox<T>* fg_box = fg_boxes.GetBBox(i);
    //polys to mask wrt box
    T w = fg_box->width();
    T h = fg_box->height();
    char* mask;
    FOR_RANGE(int32_t, poly_index, 0, polys.polygons_size()){
      FOR_RANGE(int32_t, k, 0, polys.polygons(poly_index).value_size()){
        if(k % 2 == 0){
          poly.add_value(
            (polys.polygons(poly_index).value(k)- fg_box->x1())* M / w );
        }else{
          poly.add_value(
            (polys.polygons(poly_index).value(k)- fg_box->y1())* M / h );
        }
      }
      DoubleList poly;
      char * mask_k;
      RLE rle;//to do : include coco api.h
      rleFrPoly(rle, poly, polys.polygons(poly_index).value_size(), M, M);
      rleDecode(rle, mask_k, 1);
      if(poly_index == 0){mask = mask_k;}
      mask = mask_k | mask;
    }
  }
  
 

  //output masks
}

template<DeviceType device_type, typename T>
void MaskTargetKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaskTargetConf, MaskTargetKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
