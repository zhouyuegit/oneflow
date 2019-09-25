#ifndef ONEFLOW_CORE_KERNEL_ARC_FACE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ARC_FACE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ArcFaceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArcFaceKernel);
  ArcFaceKernel() = default;
  ~ArcFaceKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  int32_t lower_bound_;
};

template<DeviceType device_type, typename T, typename K>
struct ArcFaceKernelUtil final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const T* in, const K* label, const int64_t lower_bound, const T cos_m,
                      const T sin_m, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const T* out_diff, const K* label, const int64_t lower_bound, const T cos_m,
                       const T sin_m, const T* sin_theta_data, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARC_FACE_KERNEL_H_
