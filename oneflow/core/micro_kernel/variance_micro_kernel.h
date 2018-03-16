#ifndef ONEFLOW_CORE_MICRO_KERNEL_VARIANCE_MICRO_KERNEL_H_
#define ONEFLOW_CORE_MICRO_KERNEL_VARIANCE_MICRO_KERNEL_H_

#include "oneflow/core/micro_kernel/micro_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class VarianceMicroKernel final
    : public MicroKernelIf<VarianceMicroKernel, device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VarianceMicroKernel);
  VarianceMicroKernel(BlobSymbol* input, BlobSymbol* mean, float epsilon,
                      const std::string& out_blob_name,
                      const std::string& out_diff_blob_name)
      : MicroKernelIf<VarianceMicroKernel, device_type, T>(
            {input, mean}, out_blob_name, out_diff_blob_name),
        input_(input),
        mean_(mean) {}
  ~VarianceMicroKernel() override {}

  //  Getters
  const BlobSymbol& input() const { return *input_; }
  const BlobSymbol& mean() const { return *mean_; }

 private:
  const BlobSymbol* input_;
  const BlobSymbol* mean_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_VARIANCE_MICRO_KERNEL_H_
