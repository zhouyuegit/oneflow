#ifndef ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/micro_kernel/micro_kernel_graph.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalizationKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationKernel);
  NormalizationKernel() = default;
  ~NormalizationKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override;
  
 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<MicroKernelGraph> micro_kernel_graph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
