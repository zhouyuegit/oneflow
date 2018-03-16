#ifndef ONEFLOW_CORE_MICRO_KERNEL_MEAN_MICRO_KERNEL_H_
#define ONEFLOW_CORE_MICRO_KERNEL_MEAN_MICRO_KERNEL_H_

#include "oneflow/core/micro_kernel/micro_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MeanMicroKernel final
    : public MicroKernelIf<MeanMicroKernel, device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MeanMicroKernel);
  MeanMicroKernel(BlobSymbol* input, const std::string& out_blob_name,
                  const std::string& out_diff_blob_name)
      : MicroKernelIf<MeanMicroKernel, device_type, T>({input}, out_blob_name,
                                                       out_diff_blob_name),
        input_(input) {}
  ~MeanMicroKernel() override {}

 private:
  const BlobSymbol* input_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_MEAN_MICRO_KERNEL_H_
