#include "oneflow/core/kernel/gelu_kernel.h"
#include "oneflow/core/kernel/gelu_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void GeluGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  GeluKernelUtil<device_type, T>::GeluBackward(ctx.device_ctx, x_blob->static_shape().elem_cnt(),
                                               x_blob->dptr<T>(), BnInOp2Blob("dy")->dptr<T>(),
                                               BnInOp2Blob("dx")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& GeluGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gelu_grad_conf();
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kGeluGradConf, GeluGradKernel);

class GeluHalfGpuGradKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluHalfGpuGradKernel);
  GeluHalfGpuGradKernel() = default;
  ~GeluHalfGpuGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override {
    LOG(INFO) << "GeluHalfGpuGradKernel";
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_conf(); }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGeluGradConf, DeviceType::kGPU, float16,
                                      GeluHalfGpuGradKernel);
}  // namespace oneflow
