#include "oneflow/core/kernel/yolo_prob_loss_kernel.h"

namespace oneflow {

template<typename T>
void YoloProbLossKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* prob_loss_blob = BnInOp2Blob("prob_loss");
  std::memset(prob_loss_blob->mut_dptr(), 0, prob_loss_blob->shape().elem_cnt() * sizeof(T));
  const Blob* prob_logistic_blob = BnInOp2Blob("prob_logistic");
  FOR_RANGE(int32_t, im_i, 0, prob_logistic_blob->shape().At(0)) {
    CalcProbLoss(im_i, BnInOp2Blob);
  }
}

template<typename T>
void YoloProbLossKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // const Blob* prob_loss_diff_blob = BnInOp2Blob(GenDiffBn("prob_loss"));
  const Blob* prob_diff_tmp_blob = BnInOp2Blob("prob_diff_tmp");
  Blob* prob_logistic_diff_blob = BnInOp2Blob(GenDiffBn("prob_logistic"));
  prob_logistic_diff_blob->CopyDataContentFrom(ctx.device_ctx, prob_diff_tmp_blob);
}

template<typename T>
void YoloProbLossKernel<T>::CalcProbLoss(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* prob_logistic_blob = BnInOp2Blob("prob_logistic");
  const T* prob_logistic_ptr = prob_logistic_blob->dptr<T>(im_index);
  const int32_t* pos_cls_label_ptr = BnInOp2Blob("pos_cls_label")->dptr<int32_t>(im_index);
  const int32_t* pos_inds_ptr = BnInOp2Blob("pos_inds")->dptr<int32_t>(im_index);
  const int32_t* neg_inds_ptr = BnInOp2Blob("neg_inds")->dptr<int32_t>(im_index);
  // const size_t pos_num = BnInOp2Blob("pos_inds")->dim1_valid_num(im_index);
  // const size_t neg_num = BnInOp2Blob("neg_inds")->dim1_valid_num(im_index);
  T* prob_loss_ptr = BnInOp2Blob("prob_loss")->mut_dptr<T>(im_index);
  T* prob_diff_tmp_ptr = BnInOp2Blob("prob_diff_tmp")->mut_dptr<T>(im_index);
  Blob* label_tmp_blob = BnInOp2Blob("label_tmp");
  int32_t* label_tmp_ptr = label_tmp_blob->mut_dptr<int32_t>();
  const int32_t num_prob = 1 + op_conf().yolo_prob_loss_conf().num_classes();
  FOR_RANGE(size_t, i, 0, prob_logistic_blob->shape().At(1)) {
    if (pos_inds_ptr[i] == -1) { break; }
    std::memset(label_tmp_blob->mut_dptr(), 0,
                label_tmp_blob->shape().elem_cnt() * sizeof(int32_t));
    const int32_t box_index = pos_inds_ptr[i];
    label_tmp_ptr[0] = 1;
    if (pos_cls_label_ptr[box_index] >= 0) {
      label_tmp_ptr[pos_cls_label_ptr[box_index] + 1] = 1;  //(obj_prob, 80cls_prob)
    }
    CalSub(num_prob, label_tmp_ptr, prob_logistic_ptr + num_prob * box_index,
           prob_diff_tmp_ptr + num_prob * box_index);
  }

  FOR_RANGE(size_t, i, 0, prob_logistic_blob->shape().At(1)) {
    if (neg_inds_ptr[i] == -1) { break; }
    const int32_t box_index = neg_inds_ptr[i];
    prob_diff_tmp_ptr[box_index * num_prob] = 0 - prob_logistic_ptr[box_index * num_prob];
    prob_loss_ptr[box_index] = -1;  // todo: compute prob loss, now set -1
  }
}

template<typename T>
void YoloProbLossKernel<T>::CalSub(const int32_t n, const int32_t* label_ptr, const T* pred_ptr,
                                   T* loss_ptr) const {
  for (int64_t i = 0; i < n; ++i) { loss_ptr[i] = label_ptr[i] - pred_ptr[i]; }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloProbLossConf, YoloProbLossKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
