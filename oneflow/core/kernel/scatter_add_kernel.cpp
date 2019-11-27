#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class ScatterAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScatterAddKernel);
  ScatterAddKernel() = default;
  ~ScatterAddKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
const PbMessage& ScatterAddKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().scatter_add_conf();
}

template<DeviceType device_type, typename T, typename K>
void ScatterAddKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* updates = BnInOp2Blob("updates");
  Blob* ref = BnInOp2Blob("ref");
  const int64_t offset = this->kernel_conf().scatter_add_conf().lower_bound();
  CHECK_EQ(this->kernel_conf().scatter_add_conf().upper_bound() - offset, ref->shape().At(0));
  GatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices, updates, 0, ref, offset);
}

#define MAKE_SCATTER_ADD_KERNEL_ENTRY(device_type_v, data_type_pair, indices_type_pair) \
  NEW_REGISTER_KERNEL(OperatorConf::kScatterAddConf,                                    \
                      ScatterAddKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(indices_type_pair)>)            \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                     \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)   \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())     \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                \
                    == kernel_conf.scatter_add_conf().indices_data_type()));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_SCATTER_ADD_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_SCATTER_ADD_KERNEL_ENTRY

}  // namespace oneflow
