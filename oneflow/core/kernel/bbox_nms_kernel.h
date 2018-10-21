#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BboxNmsKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxNmsKernel);
  BboxNmsKernel() = default;
  ~BboxNmsKernel() = default;

  using BBox = BBoxImpl<T, BBoxBase, BBoxCoord::kCorner>;

 private:
  using Image2IndexVecMap = HashMap<int32_t, std::vector<int32_t>>;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  Image2IndexVecMap GroupBBox(Blob* target_bbox_blob) const;
  std::vector<int32_t> Nms(const std::vector<int32_t>& bbox_row_ids, const Blob* bbox_prob_blob,
                           Blob* bbox_score_blob, Blob* bbox_out_blob) const;

  void OutputBBox(const std::vector<int32_t> out_bbox_inds, const Blob* target_bbox_blob,
                  Blob* out_bbox_blob) const;
  void OutputBBoxScore(const std::vector<int32_t> out_bbox_inds, const Blob* bbox_score_blob,
                       Blob* out_bbox_score_blob) const;
  void OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds, const int32_t num_classes,
                       Blob* out_bbox_label_blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_
