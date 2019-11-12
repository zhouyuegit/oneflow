#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class DecodeOneRecKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOneRecKernel);
  DecodeOneRecKernel() = default;
  ~DecodeOneRecKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

void DecodeOneRecKernel::Forward(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const std::string& bn : this->op_attribute().output_bns()) {
    Blob* blob = BnInOp2Blob(bn);
    Memset<DeviceType::kCPU>(ctx.device_ctx, blob->mut_dptr(), 0,
                             blob->ByteSizeOfDataContentField());
  }
}

REGISTER_KERNEL(OperatorConf::kDecodeOnerecConf, DecodeOneRecKernel);

}  // namespace oneflow
