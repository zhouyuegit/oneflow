#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class NopKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NopKernel);
  NopKernel() = default;
  ~NopKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().nop_conf(); }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNopConf, NopKernel);

}  // namespace oneflow
