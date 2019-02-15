#ifndef ONEFLOW_CORE_KERNEL_YOLO_BOX_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_YOLO_BOX_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class YoloBoxLossKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloBoxLossKernel);
  YoloBoxLossKernel() = default;
  ~YoloBoxLossKernel() = default;

  using BBox = BBoxImpl<const T, BBoxCategory::kXYWH>;
  using BoxesSlice = BBoxIndices<IndexSequence, BBox>;
  using BoxesWithMaxOverlapSlice = MaxOverlapIndices<BoxesSlice>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ClearOutputBlobs(const KernelCtx& ctx,
                        const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesWithMaxOverlapSlice CalcBoxesAndGtBoxesMaxOverlaps(
      int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void CalcSamplesAndBboxLoss(const int64_t im_index, BoxesWithMaxOverlapSlice& boxes,
                              const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void CalcBboxLoss(const int64_t im_index, const BoxesWithMaxOverlapSlice& boxes,
                    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void BboxCoordinateTransform(const int32_t box_index, BBox* pred_box) const;
  void BboxCoordinateTransformInverse(const int32_t box_index, BBox* truth_box) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_YOLO_BOX_LOSS_KERNEL_H_
