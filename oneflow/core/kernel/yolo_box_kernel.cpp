#include "oneflow/core/kernel/yolo_box_kernel.h"

namespace oneflow {

template<typename T>
void YoloBoxKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");          //(n,hw3,4) shape constant
  const Blob* probs_blob = BnInOp2Blob("probs");        //(n,hw3,81) shape constant
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");        //(n,hw3,4) shape variable
  Blob* out_probs_blob = BnInOp2Blob("out_probs");      //(n,hw3,4) shape variable
  Blob* probs_index_blob = BnInOp2Blob("probs_index");  // (h*w*3)
  FOR_RANGE(int64_t, i, 0, bbox_blob->shape().At(0)) {
    int32_t* probs_idx_ptr = probs_index_blob->mut_dptr<int32_t>();
    const T* bbox_ptr = bbox_blob->dptr<T>(i);
    const T* probs_ptr = probs_blob->dptr<T>(i);
    T* out_bbox_ptr = out_bbox_blob->mut_dptr<T>(i);
    T* out_probs_ptr = out_probs_blob->mut_dptr<T>(i);
    IndexSequence index_slice(bbox_blob->shape().At(1), probs_idx_ptr, true);
    FilterAndSetProbs(probs_ptr, index_slice, out_probs_ptr);
    WriteOutBBox(bbox_ptr, index_slice, out_bbox_ptr);
  }
}

template<typename T>
void YoloBoxKernel<T>::WriteOutBBox(const T* bbox_ptr, IndexSequence& index_slice,
                                    T* out_bbox_ptr) const {
  const YoloBoxOpConf& conf = op_conf().yolo_box_conf();
  int new_w = 0;
  int new_h = 0;
  if (((float)conf.image_width() / conf.image_origin_width())
      < ((float)conf.image_height() / conf.image_origin_height())) {
    new_w = conf.image_width();
    new_h = (conf.image_origin_height() * conf.image_width()) / conf.image_origin_width();
  } else {
    new_h = conf.image_height();
    new_w = (conf.image_origin_width() * conf.image_height()) / conf.image_origin_height();
  }
  FOR_RANGE(size_t, i, 0, index_slice.size()) {
    int32_t index = index_slice.GetIndex(i);
    int32_t iw = (index / conf.nbox()) % conf.layer_width();
    int32_t ih = (index / conf.nbox()) / conf.layer_width();
    int32_t ibox = index % conf.nbox();
    float box_x = (bbox_ptr[index * 4 + 0] + iw) / conf.layer_width();
    float box_y = (bbox_ptr[index * 4 + 1] + ih) / conf.layer_height();
    float box_w = exp(bbox_ptr[index * 4 + 2]) * conf.biases(2 * ibox) / conf.image_width();
    float box_h = exp(bbox_ptr[index * 4 + 3]) * conf.biases(2 * ibox + 1) / conf.image_height();
    out_bbox_ptr[i * 4 + 0] = (box_x - (conf.image_width() - new_w) / 2.0 / conf.image_width())
                              / ((float)new_w / conf.image_width());
    out_bbox_ptr[i * 4 + 1] = (box_y - (conf.image_height() - new_h) / 2.0 / conf.image_height())
                              / ((float)new_h / conf.image_height());
    out_bbox_ptr[i * 4 + 2] = box_w * (float)conf.image_width() / new_w;
    out_bbox_ptr[i * 4 + 3] = box_h * (float)conf.image_height() / new_h;
  }
}

template<typename T>
void YoloBoxKernel<T>::FilterAndSetProbs(const T* probs_ptr, IndexSequence& index_slice,
                                         T* out_probs_ptr) const {
  const YoloBoxOpConf& conf = op_conf().yolo_box_conf();
  const int32_t num_classes = conf.num_classes();
  const int32_t num_probs = num_classes + 1;
  index_slice.Filter(
      [&](int32_t index) { return probs_ptr[index * num_probs] <= conf.prob_thresh(); });
  FOR_RANGE(size_t, i, 0, index_slice.size()) {
    int32_t index = index_slice.GetIndex(i);
    const T obj_prob = probs_ptr[index * num_probs];
    out_probs_ptr[i * num_probs] = obj_prob;
    FOR_RANGE(size_t, j, 1, num_classes + 1) {
      T cls_prob = probs_ptr[index * num_probs + j] * obj_prob;
      out_probs_ptr[i * num_probs + j] = cls_prob > conf.prob_thresh() ? cls_prob : 0;
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloBoxConf, YoloBoxKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
