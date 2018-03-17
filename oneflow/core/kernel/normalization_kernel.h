#ifndef ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalizationKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationKernel);
  NormalizationKernel() = default;
  ~NormalizationKernel() = default;

 private:
  void InitModelBlobsWithDir(
      DeviceCtx* ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithOpConf(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void CalcMeanAndVariance(
      const KernelCtx&, const std::function<Blob*(const std::string&)>&) const;

  void UpdateMovingMeanAndMovingVariance(
      const KernelCtx&, const std::function<Blob*(const std::string&)>&) const;

  void Normalize(const KernelCtx&,
                 const std::function<Blob*(const std::string&)>&,
                 const Blob* mean_blob, const Blob* variance_blob) const;

  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct NormalizationKernelUtil final {
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x,
                    const float epsilon, T* y) {
    KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
    KernelUtil<device_type, T>::Rsqrt(ctx, n, y, epsilon);
  }
  static void Scal(DeviceCtx* ctx, const int64_t n, const T* x, const T* scal,
                   T* y) {
    KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
    KernelUtil<device_type, T>::Scal(ctx, n, scal, y, 1);
  }
  static void ScalarSub(DeviceCtx* ctx, const int64_t n, const T* x,
                        const T* scalar_ptr, T* y) {
    KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
    KernelUtil<device_type, T>::Axpy(ctx, n, static_cast<T>(-1), scalar_ptr, 0,
                                     y, 1);
  }
  static void ScalarAdd(DeviceCtx* ctx, const int64_t n, const T* x,
                        const T* scalar_ptr, T* y) {
    KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
    KernelUtil<device_type, T>::Axpy(ctx, n, static_cast<T>(1), scalar_ptr, 0,
                                     y, 1);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
