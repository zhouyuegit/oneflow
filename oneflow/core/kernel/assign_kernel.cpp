#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class AssignKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignKernel);
  AssignKernel() = default;
  ~AssignKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void AssignKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->kernel_conf().op_attribute().op_conf().name() == "Primary-LR-State-Assign"
      || this->kernel_conf().op_attribute().op_conf().name() == "Secondary-LR-State-Assign") {
    const float* ref_ptr = BnInOp2Blob("ref")->dptr<float>();
    const float* value_ptr = BnInOp2Blob("value")->dptr<float>();
    float ref = -1;
    float value = -1;
    MemoryCase mem_case{};
    mem_case.mutable_host_mem();
    AutoMemcpy(ctx.device_ctx, &ref, ref_ptr, BnInOp2Blob("ref")->ByteSizeOfBlobBody(), mem_case,
               BnInOp2Blob("ref")->mem_case());
    AutoMemcpy(ctx.device_ctx, &value, value_ptr, BnInOp2Blob("value")->ByteSizeOfBlobBody(),
               mem_case, BnInOp2Blob("value")->mem_case());
    LOG(INFO) << this->kernel_conf().op_attribute().op_conf().name() << " before : ref " << ref
              << " value " << value << " op_name ";
  }
  BnInOp2Blob("ref")->CopyValidDataContentFrom(ctx.device_ctx, BnInOp2Blob("value"));
  if (this->kernel_conf().op_attribute().op_conf().name() == "Primary-LR-State-Assign"
      || this->kernel_conf().op_attribute().op_conf().name() == "Secondary-LR-State-Assign") {
    const float* ref_ptr = BnInOp2Blob("ref")->dptr<float>();
    const float* value_ptr = BnInOp2Blob("value")->dptr<float>();
    float ref = -1;
    float value = -1;
    MemoryCase mem_case{};
    mem_case.mutable_host_mem();
    AutoMemcpy(ctx.device_ctx, &ref, ref_ptr, BnInOp2Blob("ref")->ByteSizeOfBlobBody(), mem_case,
               BnInOp2Blob("ref")->mem_case());
    AutoMemcpy(ctx.device_ctx, &value, value_ptr, BnInOp2Blob("value")->ByteSizeOfBlobBody(),
               mem_case, BnInOp2Blob("value")->mem_case());
    LOG(INFO) << this->kernel_conf().op_attribute().op_conf().name() << " after : ref " << ref
              << " value " << value << " op_name ";
  }
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kCPU,
                            AssignKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kGPU,
                            AssignKernel<DeviceType::kGPU>);

}  // namespace oneflow
