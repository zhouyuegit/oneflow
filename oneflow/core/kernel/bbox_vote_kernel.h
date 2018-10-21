#ifndef ONEFLOW_CORE_KERNEL_BBOX_VOTE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BBOX_VOTE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class Scoring {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Scoring);
  Scoring() = default;

  using BBox = BBoxImpl<T, ImIndexedBBoxBase, BBoxCoord::kCorner>;
  using ScoredBoxesIndex = ScoreIndices<BBoxIndices<IndexSequence, BBox>, T>;

  void Init(const BboxVoteConf& vote_conf, ScoringMethod method_type) {
    vote_conf_ = vote_conf;
    method_type_ = method_type;
  }

  const BboxVoteConf& conf() const { return vote_conf_; }
  T scoring(const ScoredBoxesIndex<T>&, const T default_score,
            const std::function<void(const std::function<void(int32_t, float)>&)>&) const;

 private:
  template<typename F>
  T Avg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const;

  template<typename F>
  T IouAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const;

  template<typename F>
  T GeneralizedAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const;

  template<typename F>
  T QuasiSum(const ScoredBoxesIndex<T>& slice, const F& ForEach) const;

  template<typename F>
  T TempAvg(const ScoredBoxesIndex<T>& slice, const F& ForEach) const;

  BboxVoteConf vote_conf_;
  ScoringMethod method_type_;
};

template<DeviceType device_type, typename T>
class BboxVoteKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxVoteKernel);
  BboxVoteKernel() = default;
  ~BboxVoteKernel() = default;

  using BBox = BBoxImpl<T, BBoxBase, BBoxCoord::kCorner>;
  using ScoredBoxesIndices = ScoreIndices<BBoxIndices<IndexSequence, BBox>, T>;

 private:
  void VirtualKernelInit(const ParallelContext* parallel_ctx) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ApplyBboxAndScore(const std::vector<int32_t>& bbox_row_ids, const Blob* bbox_blob,
                         const Blob* bbox_score_blob, Blob* bbox_out_blob) const;
  void VoteBboxAndScore(const ScoredBoxesIndices& pre_vote_inds,
                        ScoredBoxesIndices& post_vote_inds) const;
  void VoteBbox(const ScoredBoxesIndices& pre_vote_inds, BBox* voted_bbox,
                const std::function<void(const std::function<void(int32_t, float)>&)>&) const;
  void OutputBBox(const std::vector<int32_t> out_bbox_inds, const Blob* target_bbox_blob,
                  Blob* out_bbox_blob) const;
  void OutputBBoxScore(const std::vector<int32_t> out_bbox_inds, const Blob* bbox_score_blob,
                       Blob* out_bbox_score_blob) const;
  void OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds, const int32_t num_classes,
                       Blob* out_bbox_label_blob) const;

  Scoring<T> scoring_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_VOTE_KERNEL_H_
