#include "oneflow/core/kernel/mask_target_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace{

  auto GetValidDim1 = [](const Blob* blob, int32_t no) -> int32_t {
      if (blob->dim1_valid_num() != nullptr) { return blob->dim1_valid_num(no); }
      return blob->shape().At(1);
  };

  auto GetValidDim2 = [](const Blob* blob, int32_t no) -> int32_t {
      if (blob->dim2_valid_num() != nullptr) { return blob->dim2_valid_num(no); }
      return blob->shape().At(2);
  };

}
template<DeviceType device_type, typename T>
void MaskTargetKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(int64_t, i, 0, BnInOp2Blob("sample_rois")->shape().At(0)) {
    auto mask_boxes = GetMaskBoxes(i, BnInOp2Blob);
    auto fg_boxes = GetFgBoxes(i, BnInOp2Blob);
    ComputeFgBoxesAndMaskBoxesOverlaps(mask_boxes, fg_boxes);
    Polys2MaskWrtBox(fg_boxes, BnInOp2Blob);
  }
}

template<typename T>
MaskBoxes MaskTargetKernel<device_type, T>::GetMaskBoxes(size_t im_index, 
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const{
  FloatList16 mask_boxes; 
  int32_t valid_polys_num = GetValidDim1(seg_polys, im_index);
  FOR_RANGE(int32_t, gt_index, 0, valid_polys_num){
    int32_t valid_polys_length = GetValidDim2(seg_polys, im_index, gt_index);
    PolygonList polys;
    polys.ParseFromArray(BnInOp2Blob("seg_polys")->dptr<char>(im_index, gt_index), valid_polys_length);
    float x0 = 0;
    float x1 = 0;
    float y0 = 0;
    float y1 = 0;
    FOR_RANGE(int32_t, poly_index, 0, polys.polygons_size()){
      FOR_RANGE(int32_t, k, 0, polys.polygons(poly_index).value().value_size()){
        if(k % 2 == 0){
          x0 = std::min(x0, polys.polygons(poly_index).value().value(k))
          x1 = std::max(x1, polys.polygons(poly_index).value().value(k)) 
        }else{
          y0 = std::min(x0, polys.polygons(poly_index).value().value(k))
          y1 = std::max(x1, polys.polygons(poly_index).value().value(k)) 
        }
      }
    }
    mask_boxes.add_value(x0);
    mask_boxes.add_value(x1);
    mask_boxes.add_value(y0);
    mask_boxes.add_value(y1); 
  }
  MaskBoxes mask_boxes(*mask_boxes)
  return mask_boxes; 
}

template<typename T>
BoxesWithMaxOverlap MaskTargetKernel<device_type, T>::GetFgBoxes(size_t im_index, 
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const{
  
  
}

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
