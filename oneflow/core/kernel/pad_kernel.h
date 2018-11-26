#ifndef ONEFLOW_CORE_KERNEL_PAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PadKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadKernel);
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

};

template<DeviceType device_type, typename T>
struct PadKernelUtil {
  static void Forward(const KernelCtx& ctx, const int64_t elem_cnt, const int64_t num_axes,
                      const int64_t* outshape_count,const int64_t* outshape_at,
                      const int64_t* inshape_count,const int64_t* inshape_at,
                      const T* in_dptr, T* out_dptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PAD_KERNEL_H_