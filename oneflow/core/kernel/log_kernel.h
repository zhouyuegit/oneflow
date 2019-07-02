#ifndef ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LogKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogKernel);
  LogKernel() = default;
  ~LogKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};


template<DeviceType device_type, typename T>
struct LogKernelUtil {
  static void Log(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_
