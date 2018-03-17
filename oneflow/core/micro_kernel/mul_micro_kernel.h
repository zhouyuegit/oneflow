#ifndef ONEFLOW_CORE_MICRO_KERNEL_MUL_MICRO_KERNEL_H_
#define ONEFLOW_CORE_MICRO_KERNEL_MUL_MICRO_KERNEL_H_

#include "oneflow/core/micro_kernel/micro_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MulMicroKernel final
    : public MicroKernelIf<MulMicroKernel, device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MulMicroKernel);
  MulMicroKernel(BlobSymbol* a_blob_symbol, BlobSymbol* b_blob_symbol,
                 BlobSymbol* out_blob_symbol)
      : MicroKernelIf<MulMicroKernel, device_type, T>(
            {a_blob_symbol, b_blob_symbol}, out_blob_symbol),
        a_blob_symbol_(a_blob_symbol),
        b_blob_symbol_(b_blob_symbol) {}
  ~MulMicroKernel() override {}

  //  Getters
  const BlobSymbol& a_blob_symbl() const { return *a_blob_symbol_; }
  const BlobSymbol& b_blob_symbl() const { return *b_blob_symbol_; }

 private:
  const BlobSymbol* a_blob_symbol_;
  const BlobSymbol* b_blob_symbol_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_MUL_MICRO_KERNEL_H_
