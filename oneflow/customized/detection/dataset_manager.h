#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATASET_MANAGER_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATASET_MANAGER_H_

#include "oneflow/customized/detection/dataset.h"

namespace oneflow {

namespace detection {

class DatasetManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DatasetManager);
  DatasetManager() = default;
  ~DatasetManager() = default;
  std::shared_ptr<Dataset> Get(const std::string& dataset_name);
  std::shared_ptr<Dataset> GetOrCreateDataset(const DetectionDatasetProto& proto);

 private:
  HashMap<std::string, std::shared_ptr<Dataset>> dataset_map_;
  std::mutex mtx_;
};

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATASET_MANAGER_H_
