#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOADER_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOADER_H_

// #include "oneflow/core/data/dataset.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/customized/detection/data_sampler.h"
#include <thread>

namespace oneflow {

namespace detection {

class DataInstance;

class DataLoader final {
 public:
  using BatchDataInstance = std::vector<DataInstance>;

  OF_DISALLOW_COPY_AND_MOVE(DataLoader);
  DataLoader(const DetectionDataLoadOpConf& op_conf,
             const DetectionDataLoadKernelConf& kernel_conf);
  ~DataLoader();

  std::shared_ptr<BatchDataInstance> FetchBatch();

 protected:
  void LoadBatch();
  void Close();

 private:
  DetectionDataLoadOpConf op_conf_;
  DetectionDataLoadKernelConf kernel_conf_;

  std::shared_ptr<Dataset> dataset_;
  DataSamplerContext sampler_ctx_;

  Buffer<std::shared_ptr<BatchDataInstance>> batch_buffer_;

  std::atomic<bool> is_closed_;
  std::thread load_thrd_;
};

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOADER_H_
