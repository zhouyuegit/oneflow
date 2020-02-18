#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarAddByTensorKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddByTensorKernel);
  ScalarAddByTensorKernel() = default;
  ~ScalarAddByTensorKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_add_by_tensor_conf();
  }
};

template<DeviceType device_type, typename T>
void ScalarAddByTensorKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* scalar_blob = BnInOp2Blob("scalar");
  Blob* out_blob = BnInOp2Blob("out");
  NewKernelUtil<device_type>::AddByScalarPtr(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                             in_blob->dptr<T>(), scalar_blob->dptr<T>(),
                                             out_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kScalarAddByTensorConf, ScalarAddByTensorKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
