#ifndef ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class OneHotKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneHotKernel)
  OneHotKernel() = default;
  ~OneHotKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  int32_t lower_bound_;
  int32_t upper_bound_;
};

template<DeviceType device_type, typename T, typename K>
struct OneHotKernelUtil final {
  static void Encode(DeviceCtx* ctx, const K* indices, int64_t num_indices, int64_t lower_bound,
      int64_t upper_bound, T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_
