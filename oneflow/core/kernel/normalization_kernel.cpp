#include "oneflow/core/kernel/normalization_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* beta = BnInOp2Blob("beta");
  const Blob* gamma = BnInOp2Blob("gamma");
  Blob* moving_mean = BnInOp2Blob("moving_mean");
  Blob* moving_variance = BnInOp2Blob("moving_variance");
  Blob* out_blob = BnInOp2Blob("out");
  if (JobDesc::Singleton()->IsTrain()) {

  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf, NormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
