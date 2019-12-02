#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class OnesLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OnesLikeKernel);
  OnesLikeKernel() : is_init_(false) {}
  ~OnesLikeKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;

  mutable bool is_init_;
};

template<DeviceType device_type, typename T>
void OnesLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (is_init_) { return; }
  InitializerConf initializer_conf;
  initializer_conf.mutable_constant_int_conf()->set_value(1);
  KernelUtil<device_type, T>::InitializeWithConf(ctx.device_ctx, initializer_conf, 0,
                                                 BnInOp2Blob("out"));
  is_init_ = true;
}

template<DeviceType device_type, typename T>
const PbMessage& OnesLikeKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().ones_like_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kOnesLikeConf, OnesLikeKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
