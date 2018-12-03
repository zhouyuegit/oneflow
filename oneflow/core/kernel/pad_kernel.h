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
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PadKernelUtil {
  static void Forward(const KernelCtx& ctx, 
                      const PbRf<int32_t>& padding_before, const PbRf<int32_t>& padding_after,
                      int32_t* outshape_count, int32_t* inshape_count,
                      int32_t* padding_left_bound, int32_t* padding_right_bound, 
                      const Blob* in_blob, Blob* out_blob);
          
  static void Backward(const KernelCtx& ctx,
                       const int32_t* outshape_count, const int32_t* inshape_count,
                       const int32_t* padding_left_bound, const int32_t* padding_right_bound,
                       Blob* in_diff_blob, const Blob* out_diff_blob);

};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PAD_KERNEL_H_