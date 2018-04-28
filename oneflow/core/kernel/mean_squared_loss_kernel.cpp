#include "oneflow/core/kernel/mean_squared_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void MeanSquaredLossKernel<device_type, PredType, LabelType>::
    VirtualLossForwardDataContent(
        const KernelCtx& ctx,
        std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* pred_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* loss_blob = BnInOp2Blob("loss");
  Blob* diff_blob = BnInOp2Blob("diff");
  int64_t inst_num = pred_blob->shape().At(0);
  int64_t label_dim = label_blob->shape().Count(1);
  MeanSquaredLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, inst_num, label_dim, label_blob->dptr<LabelType>(),
      pred_blob->dptr<PredType>(), diff_blob->mut_dptr<PredType>(),
      loss_blob->mut_dptr<PredType>());

  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  if (pred_diff_blob != nullptr) {
    Memset<device_type>(ctx.device_ctx, pred_diff_blob->mut_dptr<PredType>(), 0,
                        pred_diff_blob->TotalByteSize());
    MeanSquaredLossKernelUtil<device_type, PredType, LabelType>::Backward(
        ctx.device_ctx, inst_num, label_dim, diff_blob->dptr<PredType>(),
        pred_diff_blob->mut_dptr<PredType>());
  }
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
MeanSquaredLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.mean_squared_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct MeanSquaredLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t inst_num,
                      const int64_t label_dim, const LabelType* label,
                      const PredType* pred, PredType* diff, PredType* loss) {
    const int64_t n = inst_num * label_dim;
    for (int64_t i = 0; i < n; ++i) {
      diff[i] = static_cast<PredType>(label[i]);
    }
    KernelUtil<DeviceType::kCPU, PredType>::Axpy(
        ctx, n, static_cast<const PredType>(-1), pred, 1, diff, 1);
    for (int64_t i = 0; i < inst_num; ++i) {
      KernelUtil<DeviceType::kCPU, PredType>::Dot(
          ctx, label_dim, diff, inst_num, diff, inst_num, loss + i);
    }
    const PredType mean = static_cast<PredType>(2 * label_dim);
    KernelUtil<DeviceType::kCPU, PredType>::Div(ctx, inst_num, loss, &mean);
  }

  static void Backward(DeviceCtx* ctx, const int64_t inst_num,
                       const int64_t label_dim, const PredType* diff,
                       PredType* pred_diff) {
    KernelUtil<DeviceType::kCPU, PredType>::Copy(ctx, inst_num, diff, 1,
                                                 pred_diff, 1);
    KernelUtil<DeviceType::kCPU, PredType>::Scal(
        ctx, inst_num, static_cast<const PredType>(-1), pred_diff, 1);
    const PredType mean = static_cast<PredType>(label_dim);
    KernelUtil<DeviceType::kCPU, PredType>::Div(ctx, inst_num, pred_diff,
                                                &mean);
  }
};

namespace {

Kernel* CreateMeanSquaredLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define MEAN_SQUARED_LOSS_KERNEL_ENTRY(device_type, pred_type_pair,         \
                                       label_type_pair)                     \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair),               \
              OF_PP_PAIR_SECOND(label_type_pair)),                          \
   []() {                                                                   \
     return new MeanSquaredLossKernel<device_type,                          \
                                      OF_PP_PAIR_FIRST(pred_type_pair),     \
                                      OF_PP_PAIR_FIRST(label_type_pair)>(); \
   }},

      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MEAN_SQUARED_LOSS_KERNEL_ENTRY,
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ,
                                       INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(
      kernel_conf.device_type(),
      kernel_conf.mean_squared_loss_conf().loss_conf().prediction_type(),
      kernel_conf.mean_squared_loss_conf().loss_conf().label_type()))();
}

}  // namespace

#define MAKE_ENTRY(data_type_pair, label_type_pair)       \
  template struct MeanSquaredLossKernelUtil<              \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
      OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

COMMAND(AddKernelCreator(OperatorConf::kMeanSquaredLossConf,
                         CreateMeanSquaredLossKernel));

}  // namespace oneflow
