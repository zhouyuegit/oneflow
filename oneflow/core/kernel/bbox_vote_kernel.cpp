#include "oneflow/core/kernel/bbox_vote_kernel.h"

namespace oneflow {

template<typename T>
T Scoring<T>::scoring(
    const ScoredBoxesIndex<T>& slice, const T default_score,
    const std::function<void(const std::function<void(int32_t, float)>&)>& ForEach) const {
  switch (method_type_) {
    case ScoringMethod::kId: {
      return default_score;
    }
    case ScoringMethod::kAvg: {
      return Avg(slice, ForEach);
    }
    case ScoringMethod::kIouAvg: {
      return IouAvg(slice, ForEach);
    }
    case ScoringMethod::kGeneralizedAvg: {
      return GeneralizedAvg(slice, ForEach);
    }
    case ScoringMethod::kQuasiSum: {
      return QuasiSum(slice, ForEach);
    }
    case ScoringMethod::kTempAvg: {
      return TempAvg(slice, ForEach);
    }
  }
  return {};
}

template<typename T>
template<typename F>
T Scoring<T>::Avg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const {
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    score_sum += slice.GetScore(slice_index);
    ++num;
  });
  return score_sum / num;
}

template<typename T>
template<typename F>
T Scoring<T>::IouAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const {
  T iou_weighted_score_sum = 0;
  T iou_sum = 0;
  ForEach([&](int32_t slice_index, float iou) {
    iou_weighted_score_sum += slice.GetScore(slice_index) * iou;
    iou_sum += iou;
  });
  return static_cast<T>(iou_weighted_score_sum / iou_sum);
}

template<typename T>
template<typename F>
T Scoring<T>::GeneralizedAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const {
  const float beta = vote_conf_.beta();
  T generalized_score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    generalized_score_sum += std::pow<T>(slice.GetScore(slice_index), beta);
    ++num;
  });
  return std::pow<T>(generalized_score_sum / num, 1.f / beta);
}

template<typename T>
template<typename F>
T Scoring<T>::QuasiSum(const ScoredBoxesIndex<T>& slice, const F& ForEach) const {
  const float beta = vote_conf_.beta();
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    score_sum += slice.GetScore(slice_index);
    ++num;
  });
  return static_cast<T>(score_sum / std::pow<T>(num, beta));
}

template<typename T>
template<typename F>
T Scoring<T>::TempAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const {
  // TODO
  return {};
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const BboxVoteOpConf& conf = op_conf().bbox_vote_conf();
  scoring_.Init(conf.bbox_vote(), conf.bbox_vote().scoring_method());
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* bbox_label_blob = BnInOp2Blob("bbox_label");
  const Blob* bbox_score_blob = BnInOp2Blob("bbox_score");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_bbox_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_bbox_label_blob = BnInOp2Blob("out_bbox_label");

  std::vector<int32_t> all_im_bbox_inds;
  auto im_grouped_bbox_inds = GroupBBox(bbox_blob);
  for (auto& pair : im_grouped_bbox_inds) {
    auto im_detected_bbox_inds =
        ApplyBboxAndScore(pair.second, bbox_blob, bbox_score_blob, out_bbox_blob);
    all_im_bbox_inds.insert(all_im_bbox_inds.end(), im_detected_bbox_inds.begin(),
                            im_detected_bbox_inds.end());
  }
  OutputBBox(all_im_bbox_inds, bbox_blob, out_bbox_blob);
  OutputBBoxScore(all_im_bbox_inds, bbox_score_blob, out_bbox_score_blob);
  OutputBBoxLabel(all_im_bbox_inds, bbox_prob_blob->shape().At(1), out_bbox_label_blob);
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<T>::ApplyBboxAndScore(const std::vector<int32_t>& bbox_row_ids,
                                          const Blob* bbox_blob, const Blob* bbox_score_blob,
                                          Blob* bbox_out_blob) const {
  const T* bbox_ptr = bbox_blob->dptr<T>();
  int32_t num_classes = bbox_score_blob->shape().At(1);
  std::vector<int32_t> all_cls_bbox_inds;
  all_cls_bbox_inds.reserve(bbox_row_ids.size() * num_classes);
  FOR_RANGE(int32_t, k, 1, num_classes) {
    std::vector<int32_t> cls_bbox_inds(bbox_row_ids.size());
    std::transform(bbox_row_ids.begin(), bbox_row_ids.end(), cls_bbox_inds.begin(),
                   [&](int32_t idx) { return idx * num_classes + k; });
    auto pre_vote_inds = GenScoredBoxesIndices(cls_bbox_inds.size(), cls_bbox_inds.data(),
                                               target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
    std::vector<int32_t> post_vote_bbox_inds(cls_bbox_inds.size());
    auto post_vote_inds =
        GenScoredBoxesIndices(pre_vote_inds.size(), post_vote_bbox_inds.data(),
                              target_bbox_blob->mut_dptr<T>(), bbox_score_ptr, false);
    // voting
    VoteBboxAndScore(pre_vote_inds, post_vote_inds);
    // concat all class
    all_cls_bbox_inds.insert(all_cls_bbox_inds.end(), post_vote_inds.index(),
                             post_vote_inds.index() + post_vote_inds.size());
  }
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<T>::VoteBboxAndScore(const ScoredBoxesIndices& pre_vote_inds,
                                         ScoredBoxesIndices& post_vote_inds) const {
  const T voting_thresh = op_conf().bbox_vote_conf().bbox_vote().threshold();
  FOR_RANGE(size_t, i, 0, post_vote_inds.size()) {
    const auto* votee_bbox = post_vote_inds.GetBBox(i);
    auto ForEachNearBy = [&pre_vote_inds, votee_bbox,
                          voting_thresh](const std::function<void(int32_t, float)>& Handler) {
      FOR_RANGE(size_t, j, 0, pre_vote_inds.size()) {
        const auto* voter_bbox = pre_vote_inds.GetBBox(j);
        float iou = voter_bbox->InterOverUnion(votee_bbox);
        if (iou >= voting_thresh) { Handler(j, iou); }
      }
    };
    int32_t bbox_idx = post_vote_inds.GetIndex(i);
    T* score_ptr = const_cast<T*>(post_vote_inds.score());
    score_ptr[bbox_idx] =
        scoring_method_->scoring(pre_vote_inds, post_vote_inds.GetScore(i), ForEachNearBy);
    VoteBbox(pre_vote_inds, post_vote_inds.mut_bbox(bbox_idx), ForEachNearBy);
  }
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<T>::VoteBbox(
    const ScoredBoxesIndices& pre_vote_inds, BBox* voted_bbox,
    const std::function<void(const std::function<void(int32_t, float)>&)>& ForEachNearBy) const {
  std::array<T, 4> score_weighted_bbox = {0, 0, 0, 0};
  T score_sum = 0;
  ForEachNearBy([&](int32_t voter_idx, float iou) {
    const T voter_score = pre_vote_inds.GetScore(voter_idx);
    FOR_RANGE(int32_t, k, 0, 4) {
      score_weighted_bbox[k] += pre_vote_inds.GetBBox(voter_idx)->bbox_elem(k) * voter_score;
    }
    score_sum += voter_score;
  });
  FOR_RANGE(int32_t, k, 0, 4) { voted_bbox->set_bbox_elem(k, score_weighted_bbox[k] / score_sum); }
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<device_type, T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
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
void BboxVoteKernel<T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
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
void BboxVoteKernel<T>::OutputBBoxScore(const std::vector<int32_t> out_bbox_inds,
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
void BboxVoteKernel<T>::OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds,
                                        const int32_t num_classes,
                                        Blob* out_bbox_label_blob) const {
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
void BboxVoteKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void BboxVoteKernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxVoteConf, BboxVoteKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow