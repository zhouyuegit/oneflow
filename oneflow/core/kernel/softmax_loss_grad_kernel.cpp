#include "oneflow/core/kernel/softmax_loss_grad_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void SoftmaxLossGradBackward(DeviceCtx* ctx, const int64_t lower_bound, const Blob* label,
                             Blob* in_diff) {
  SoftmaxLossGradKernelUtil<device_type, T, K>::Backward(ctx, in_diff->shape().At(0),
                                                         in_diff->shape().At(1), label->dptr<K>(),
                                                         lower_bound, in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct SoftmaxLossGradSwitchUtil final {
#define MAKE_SOFTMAX_LOSS_GRAD_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_SOFTMAX_LOSS_GRAD_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_SOFTMAX_LOSS_GRAD_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_SOFTMAX_LOSS_GRAD_STATIC_SWITCH_FUNC(SoftmaxLossGradBackward);
#undef DEFINE_SOFTMAX_LOSS_GRAD_STATIC_SWITCH_FUNC
#undef MAKE_SOFTMAX_LOSS_GRAD_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& SoftmaxLossGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().softmax_loss_grad_conf();
}

template<DeviceType device_type, typename T>
void SoftmaxLossGradKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  lower_bound_ = 0;
  if (parallel_ctx->policy() == kModelParallel) {
    auto& conf = this->op_conf().softmax_loss_grad_conf();
    BalancedSplitter splitter(conf.depth(), parallel_ctx->parallel_num());
    lower_bound_ = splitter.At(parallel_ctx->parallel_id()).begin();
  }
}

template<DeviceType device_type, typename T>
void SoftmaxLossGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("softmax"));
}

template<DeviceType device_type, typename T>
void SoftmaxLossGradKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("in_diff")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("softmax"));
  SoftmaxLossGradSwitchUtil<device_type, T>::SwitchSoftmaxLossGradBackward(
      SwitchCase(BnInOp2Blob("label")->data_type()), ctx.device_ctx, lower_bound_,
      BnInOp2Blob("label"), BnInOp2Blob(GenDiffBn("in")));
}

template<typename T, typename K>
struct SoftmaxLossGradKernelUtil<DeviceType::kCPU, T, K> final {
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const K* label, const int64_t lower_bound, T* in_diff);
};

template<typename T, typename K>
void SoftmaxLossGradKernelUtil<DeviceType::kCPU, T, K>::Backward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const K* label,
    const int64_t lower_bound, T* in_diff) {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxLossGradConf, SoftmaxLossGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
