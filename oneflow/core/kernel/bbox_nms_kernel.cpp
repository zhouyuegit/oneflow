#include "oneflow/core/kernel/bbox_nms_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BboxNmsKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* bbox_prob_blob = BnInOp2Blob("bbox_prob");
  Blob* bbox_score_blob = BnInOp2Blob("bbox_score");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_bbox_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_bbox_label_blob = BnInOp2Blob("out_bbox_label");

  std::vector<int32_t> all_im_bbox_inds;
  auto im_grouped_bbox_inds = GroupBBox(bbox_blob);
  for (auto& pair : im_grouped_bbox_inds) {
    auto im_detected_bbox_inds = Nms(pair.second, bbox_prob_blob, bbox_score_blob, out_bbox_blob);
    all_im_bbox_inds.insert(all_im_bbox_inds.end(), im_detected_bbox_inds.begin(),
                            im_detected_bbox_inds.end());
  }
  OutputBBox(all_im_bbox_inds, bbox_blob, out_bbox_blob);
  OutputBBoxScore(all_im_bbox_inds, bbox_score_blob, out_bbox_score_blob);
  OutputBBoxLabel(all_im_bbox_inds, bbox_prob_blob->shape().At(1), out_bbox_label_blob);
}

template<DeviceType device_type, typename T>
Image2IndexVecMap BboxNmsKernel<device_type, T>::GroupBBox(Blob* target_bbox_blob) const {
  Image2IndexVecMap im_grouped_bbox_inds;
  FOR_RANGE(int32_t, i, 0, target_bbox_blob->shape().At(0)) {
    const BBox* bbox = BBox::Cast(target_bbox_blob->dptr<T>(i, 0));
    int32_t im_idx = static_cast<int32_t>(bbox->im_index());
    im_grouped_bbox_inds[im_idx].emplace_back(i);
  }
  return im_grouped_bbox_inds;
}

template<DeviceType device_type, typename T>
std::vector<int32_t> BboxNmsKernel<device_type, T>::Nms(const std::vector<int32_t>& bbox_row_ids,
                                                        const Blob* bbox_prob_blob,
                                                        Blob* bbox_score_blob,
                                                        Blob* bbox_out_blob) const {
  const BboxNmsOpConf& conf = op_conf().bbox_nms_conf();
  const T* bbox_prob_ptr = bbox_prob_blob->dptr<T>();
  T* bbox_score_ptr = bbox_score_blob->mut_dptr<T>();
  int32_t num_classes = bbox_prob_blob->shape().At(1);
  std::vector<int32_t> all_cls_bbox_inds;
  all_cls_bbox_inds.reserve(bbox_row_ids.size() * num_classes);
  bool has_pre_nms_top_n = conf.has_pre_nms_top_n();
  bool has_pre_nms_threshold = conf.has_pre_nms_threshold();
  CHECK(has_top_n || has_pre_nms_threshold);
  FOR_RANGE(int32_t, k, 1, num_classes) {
    std::vector<int32_t> cls_bbox_inds(bbox_row_ids.size());
    std::transform(bbox_row_ids.begin(), bbox_row_ids.end(), cls_bbox_inds.begin(),
                   [&](int32_t idx) { return idx * num_classes + k; });
    std::sort(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t l_idx, int32_t h_idx) {
      return bbox_prob_ptr[l_idx] > bbox_prob_ptr[h_idx];
    });

    if (has_pre_nms_threshold) {
      auto lt_thresh_it =
          std::find_if(cls_bbox_inds.begin(), cls_bbox_inds.end(),
                       [&](int32_t idx) { return bbox_prob_ptr[idx] < conf.score_threshold(); });
      cls_bbox_inds.erase(lt_thresh_it, cls_bbox_inds.end());
    } else {
      if (conf.pre_nms_top_n() < cls_bbox_inds.size()) {
        cls_bbox_inds.erase(cls_bbox_inds.begin() + n, cls_bbox_inds.end());
      }
    }

    // nms
    size_t post_topn = std::min(conf.post_nms_top_n(), cls_bbox_inds.size());
    auto pre_nms_inds = GenScoredBoxesIndices(post_topn, cls_bbox_inds.data(),
                                              bbox_out_blob->dptr<T>(), bbox_prob_ptr, false);
    std::vector<int32_t> post_nms_bbox_inds(cls_bbox_inds.size());
    auto post_nms_inds = GenScoredBoxesIndices(post_nms_bbox_inds.size(), post_nms_bbox_inds.data(),
                                               bbox_out_blob->mut_dptr<T>(), bbox_score_ptr, false);
    BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_inds, post_nms_inds);
    all_cls_bbox_inds.insert(all_cls_bbox_inds.end(), post_nms_inds.index(),
                             post_nms_inds.index() + post_nms_inds.size());
  }

  return all_cls_bbox_inds;
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<device_type, T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
                                               const Blob* target_bbox_blob,
                                               Blob* out_bbox_blob) const {
  std::memset(out_bbox_blob->mut_dptr<T>(), 0,
              out_bbox_blob->static_shape().elem_cnt() * sizeof(T));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>()) + bbox_idx;
    auto* out_bbox = BBox::MutCast(out_bbox_blob->mut_dptr<T>()) + (out_cnt++);
    out_bbox->set_corner_coord(bbox->left(), bbox->top(), bbox->right(), bbox->bottom());
    out_bbox->set_im_index(bbox->im_index());
  }
  CHECK_LE(out_cnt, out_bbox_blob->static_shape().At(0));
  out_bbox_blob->set_dim0_valid_num(0, out_cnt);
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
                                  const Blob* target_bbox_blob, Blob* out_bbox_blob) const {
  std::memset(out_bbox_blob->mut_dptr<T>(), 0,
              out_bbox_blob->static_shape().elem_cnt() * sizeof(T));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>()) + bbox_idx;
    auto* out_bbox = BBox::MutCast(out_bbox_blob->mut_dptr<T>()) + (out_cnt++);
    out_bbox->set_corner_coord(bbox->left(), bbox->top(), bbox->right(), bbox->bottom());
    out_bbox->set_im_index(bbox->im_index());
  }
  CHECK_LE(out_cnt, out_bbox_blob->static_shape().At(0));
  out_bbox_blob->set_dim0_valid_num(0, out_cnt);
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<T>::OutputBBoxScore(const std::vector<int32_t> out_bbox_inds,
                                       const Blob* bbox_score_blob,
                                       Blob* out_bbox_score_blob) const {
  std::memset(out_bbox_score_blob->mut_dptr<T>(), 0,
              out_bbox_score_blob->static_shape().elem_cnt() * sizeof(T));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_score_blob->mut_dptr<T>()[out_cnt++] = bbox_score_blob->dptr<T>()[bbox_idx];
  }
  CHECK_LE(out_cnt, out_bbox_score_blob->static_shape().elem_cnt());
  out_bbox_score_blob->set_dim0_valid_num(0, out_cnt);
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<T>::OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds,
                                       const int32_t num_classes, Blob* out_bbox_label_blob) const {
  std::memset(out_bbox_label_blob->mut_dptr<int32_t>(), 0,
              out_bbox_label_blob->static_shape().elem_cnt() * sizeof(int32_t));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_label_blob->mut_dptr<int32_t>()[out_cnt++] = bbox_idx % num_classes;
  }
  CHECK_LE(out_cnt, out_bbox_label_blob->static_shape().elem_cnt());
  out_bbox_label_blob->set_dim0_valid_num(0, out_cnt);
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void BboxNmsKernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsConf, BboxNmsKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
