#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATASET_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/customized/detection/dataset.pb.h"
#include "oneflow/customized/detection/data_sampler.h"

namespace oneflow {

namespace detection {

class DataInstance;

class Dataset {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dataset);
  Dataset(const DetectionDatasetProto& proto);
  virtual ~Dataset() = default;

  virtual size_t Size() const = 0;
  virtual void GetData(int64_t idx, DataInstance* data) const = 0;
  virtual int64_t GetGroupId(int64_t idx) const { UNIMPLEMENTED(); }
  void SubmitSamplerContext(DataSamplerContext* ctx);
  std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx, size_t batch_size);

  const DetectionDatasetProto& proto() const { return proto_; }
  const DataSampler& sampler() const { return sampler_; }
  DataSampler* mut_sampler() { return &sampler_; }

 private:
  const DetectionDatasetProto& proto_;
  DataSampler sampler_;
};

#define DETECTION_DATASET_CASE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(DetectionDatasetProto::DatasetConfCase::kCoco)

#define REGISTER_DETECTION_DATASET_CREATOR(k, f) \
  REGISTER_CLASS_CREATOR(k, detection::Dataset, f, const DetectionDatasetProto&)

#define MAKE_DETECTION_DATASET_CREATOR_ENTRY(dataset_class, dataset_case) \
  {dataset_case,                                                          \
   [](const DetectionDatasetProto& proto) -> Dataset* { return new dataset_class(proto); }},

#define REGISTER_DETECTION_DATASET(dataset_case, dataset_derived_class)                          \
  namespace {                                                                                    \
                                                                                                 \
  Dataset* OF_PP_CAT(CreateDetectionDataset, __LINE__)(const DetectionDatasetProto& proto) {     \
    static const HashMap<DetectionDatasetProto::DatasetConfCase,                                 \
                         std::function<Dataset*(const DetectionDatasetProto& proto)>,            \
                         std::hash<int>>                                                         \
        creators = {OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_DETECTION_DATASET_CREATOR_ENTRY,       \
                                                     (dataset_derived_class),                    \
                                                     DETECTION_DATASET_CASE_SEQ)};               \
    return creators.at(proto.dataset_conf_case())(proto);                                        \
  }                                                                                              \
                                                                                                 \
  REGISTER_DETECTION_DATASET_CREATOR(dataset_case, OF_PP_CAT(CreateDetectionDataset, __LINE__)); \
  }

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATASET_H_
