#ifndef ONEFLOW_CORE_KERNEL_SOURCE_TICK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOURCE_TICK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class SourceTickKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickKernel);
  SourceTickKernel() = default;
  ~SourceTickKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().source_tick_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOURCE_TICK_KERNEL_H_