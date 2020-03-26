#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_SAMPLER_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_SAMPLER_H_

// #include <stddef.h>
// #include <cstdint>
#include "oneflow/core/common/util.h"
// #include <vector>

namespace oneflow {

namespace detection {

struct DataSamplerContext final {
  size_t num_replicas_;
  size_t rank_;
  size_t epoch_;
  size_t offset_;
  size_t count_;
};

class Dataset;

class DataSampler {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSampler);
  DataSampler() = delete;
  virtual ~DataSampler() = default;
  DataSampler(Dataset* dataset);

  void SubmitContext(DataSamplerContext* ctx);
  virtual std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx, size_t batch_size);

 protected:
  void GenNewEpochIndexSequence(size_t epoch);
  void CheckIndexSequenceRanOut(DataSamplerContext* ctx);
  const std::vector<int64_t>& AcquireGetOrGenEpochIndexSequence(size_t epoch);
  std::vector<int64_t>& GetEpochIndexSequence(size_t epoch) { return epoch2index_seq_.at(epoch); }
  const std::vector<int64_t>& GetEpochIndexSequence(size_t epoch) const {
    return epoch2index_seq_.at(epoch);
  }
  const Dataset* dataset() const { return dataset_; }

 private:
  const Dataset* dataset_;
  size_t max_count_;
  HashMap<size_t, std::vector<int64_t>> epoch2index_seq_;
  HashMap<size_t, size_t> epoch2consumed_count_;
  std::mt19937 gen_;
  std::mutex mtx_;
};

class GroupedDataSampler : public DataSampler {
 public:
  GroupedDataSampler(Dataset* dataset);
  ~GroupedDataSampler() override = default;
  virtual std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx,
                                                       size_t batch_size) override;

 private:
  std::vector<int64_t> group_ids_;
};

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_SAMPLER_H_
