#ifndef ONEFLOW_CORE_MICRO_KERNEL_RSQRT_MICRO_KERNEL_H_
#define ONEFLOW_CORE_MICRO_KERNEL_RSQRT_MICRO_KERNEL_H_

#include "oneflow/core/micro_kernel/micro_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RsqrtMicroKernel final
    : public MicroKernelIf<RsqrtMicroKernel, device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RsqrtMicroKernel);
  RsqrtMicroKernel(BlobSymbol* input, float epsilon,
                   BlobSymbol* out_blob_symbol)
      : MicroKernelIf<RsqrtMicroKernel, device_type, T>({input},
                                                        out_blob_symbol),
        input_(input) {}
  ~RsqrtMicroKernel() override {}

  //  Getters
  const BlobSymbol& input() const { return *input_; }

 private:
  const BlobSymbol* input_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_RSQRT_MICRO_KERNEL_H_
