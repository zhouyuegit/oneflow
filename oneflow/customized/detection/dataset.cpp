#include "oneflow/customized/detection/dataset.h"
#include "oneflow/customized/detection/data_sampler.h"

namespace oneflow {

namespace detection {

Dataset::Dataset(const DetectionDatasetProto& dataset_proto)
    : proto_(dataset_proto), sampler_(this) {}

void Dataset::SubmitSamplerContext(DataSamplerContext* ctx) { sampler_.SubmitContext(ctx); }

std::vector<int64_t> Dataset::FetchBatchIndexSequence(DataSamplerContext* ctx, size_t batch_size) {
  return sampler_.FetchBatchIndexSequence(ctx, batch_size);
}

}  // namespace detection

}  // namespace oneflow
