#include "oneflow/core/kernel/yolo_prob_loss_kernel.h"

namespace oneflow {

template<typename T>
void YoloProbLossKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutBlobs(ctx, BnInOp2Blob);
  FOR_RANGE(int32_t, im_i, 0, BnInOp2Blob("bbox_objness")->shape().At(0)) {
    CalcObjnessDiff(ctx, im_i, BnInOp2Blob);
    CalcClsProbDiff(ctx, im_i, BnInOp2Blob);
  }
}

template<typename T>
void YoloProbLossKernel<T>::ClearOutBlobs(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* bbox_objness_tmp_blob = BnInOp2Blob("bbox_objness_tmp");
  Blob* bbox_clsprob_tmp_blob = BnInOp2Blob("bbox_clsprob_tmp");
  Blob* bbox_objness_out_blob = BnInOp2Blob("bbox_objness_out");
  Blob* bbox_clsprob_out_blob = BnInOp2Blob("bbox_clsprob_out");
  std::memset(bbox_objness_tmp_blob->mut_dptr(), 0,
              bbox_objness_tmp_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_clsprob_tmp_blob->mut_dptr(), 0,
              bbox_clsprob_tmp_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_objness_out_blob->mut_dptr(), 0,
              bbox_objness_out_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_clsprob_out_blob->mut_dptr(), 0,
              bbox_clsprob_out_blob->shape().elem_cnt() * sizeof(T));
}

template<typename T>
void YoloProbLossKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_objness_tmp_blob = BnInOp2Blob("bbox_objness_tmp");
  const Blob* bbox_clsprob_tmp_blob = BnInOp2Blob("bbox_clsprob_tmp");

  const Blob* bbox_objness_out_diff_blob = BnInOp2Blob(GenDiffBn("bbox_objness_out"));
  Blob* bbox_objness_diff_blob = BnInOp2Blob(GenDiffBn("bbox_objness"));
  KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, bbox_objness_diff_blob->shape().elem_cnt(),
                                       bbox_objness_out_diff_blob->dptr<T>(),
                                       bbox_objness_tmp_blob->dptr<T>(),
                                       bbox_objness_diff_blob->mut_dptr<T>());

  const Blob* bbox_clsprob_out_diff_blob = BnInOp2Blob(GenDiffBn("bbox_clsprob_out"));
  Blob* bbox_clsprob_diff_blob = BnInOp2Blob(GenDiffBn("bbox_clsprob"));
  KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, bbox_clsprob_diff_blob->shape().elem_cnt(),
                                       bbox_clsprob_out_diff_blob->dptr<T>(),
                                       bbox_clsprob_tmp_blob->dptr<T>(),
                                       bbox_clsprob_diff_blob->mut_dptr<T>());
}

template<typename T>
void YoloProbLossKernel<T>::CalcObjnessDiff(
    const KernelCtx& ctx, const int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const T* bbox_objness_ptr = BnInOp2Blob("bbox_objness")->dptr<T>(im_index);
  T* bbox_objness_tmp_ptr = BnInOp2Blob("bbox_objness_tmp")->mut_dptr<T>(im_index);
  T* bbox_objness_out_ptr = BnInOp2Blob("bbox_objness_out")->mut_dptr<T>(im_index);
  const int32_t* pos_inds_ptr = BnInOp2Blob("pos_inds")->dptr<int32_t>(im_index);
  const int32_t* neg_inds_ptr = BnInOp2Blob("neg_inds")->dptr<int32_t>(im_index);
  const size_t pos_num = BnInOp2Blob("pos_inds")->dim1_valid_num(im_index);
  const size_t neg_num = BnInOp2Blob("neg_inds")->dim1_valid_num(im_index);
  FOR_RANGE(size_t, i, 0, pos_num) {
    const int32_t box_index = pos_inds_ptr[i];
    bbox_objness_tmp_ptr[box_index] = bbox_objness_ptr[box_index] - 1;
  }
  FOR_RANGE(size_t, i, 0, neg_num) {
    const int32_t box_index = neg_inds_ptr[i];
    bbox_objness_tmp_ptr[box_index] = bbox_objness_ptr[box_index] - 0;
  }
  KernelUtil<DeviceType::kCPU, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("bbox_objness_tmp")->shape().elem_cnt(), bbox_objness_tmp_ptr,
      bbox_objness_tmp_ptr, bbox_objness_out_ptr);
}

template<typename T>
void YoloProbLossKernel<T>::CalcClsProbDiff(
    const KernelCtx& ctx, const int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const T* bbox_clsprob_ptr = BnInOp2Blob("bbox_clsprob")->dptr<T>(im_index);
  const int32_t num_clsprobs = op_conf().yolo_prob_loss_conf().num_classes();
  const int32_t* pos_cls_label_ptr = BnInOp2Blob("pos_cls_label")->dptr<int32_t>(im_index);
  const int32_t* pos_inds_ptr = BnInOp2Blob("pos_inds")->dptr<int32_t>(im_index);
  const size_t pos_num = BnInOp2Blob("pos_inds")->dim1_valid_num(im_index);
  T* bbox_clsprob_tmp_ptr = BnInOp2Blob("bbox_clsprob_tmp")->mut_dptr<T>(im_index);
  T* bbox_clsprob_out_ptr = BnInOp2Blob("bbox_clsprob_out")->mut_dptr<T>(im_index);
  int32_t* label_tmp_ptr = BnInOp2Blob("label_tmp")->mut_dptr<int32_t>(im_index);
  FOR_RANGE(size_t, i, 0, pos_num) {
    std::memset(label_tmp_ptr, 0, num_clsprobs * sizeof(int32_t));
    const int32_t box_index = pos_inds_ptr[i];
    if (pos_cls_label_ptr[box_index] >= 0) { label_tmp_ptr[pos_cls_label_ptr[box_index]] = 1; }
    CalSub(num_clsprobs, label_tmp_ptr, bbox_clsprob_ptr + num_clsprobs * box_index,
           bbox_clsprob_tmp_ptr + num_clsprobs * box_index);
  }
  KernelUtil<DeviceType::kCPU, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("bbox_clsprob_tmp")->shape().elem_cnt(), bbox_clsprob_tmp_ptr,
      bbox_clsprob_tmp_ptr, bbox_clsprob_out_ptr);
}

template<typename T>
void YoloProbLossKernel<T>::CalSub(const int32_t n, const int32_t* label_ptr, const T* pred_ptr,
                                   T* diff_ptr) const {
  for (int64_t i = 0; i < n; ++i) { diff_ptr[i] = pred_ptr[i] - label_ptr[i]; }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloProbLossConf, YoloProbLossKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
