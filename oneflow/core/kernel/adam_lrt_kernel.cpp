#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class AdamLRTKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdamLRTKernel);
  AdamLRTKernel() = default;
  ~AdamLRTKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

void AdamLRTKernel::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AdamLRTOpConf& conf = this->op_conf().adam_lrt_conf();
  const int64_t train_step = *BnInOp2Blob("train_step")->dptr<int64_t>();
  const float learning_rate = *BnInOp2Blob("learning_rate")->dptr<float>();
  const float beta1 = conf.beta1();
  const float beta2 = conf.beta2();
  const float beta1_t = std::pow<double>(beta1, train_step + 1);
  const float beta2_t = std::pow<double>(beta2, train_step + 1);
  *BnInOp2Blob("out")->mut_dptr<float>() = learning_rate * sqrt(1 - (beta2_t)) / (1 - (beta1_t));
}

REGISTER_KERNEL(OperatorConf::kAdamLrtConf, AdamLRTKernel);

}  // namespace oneflow
