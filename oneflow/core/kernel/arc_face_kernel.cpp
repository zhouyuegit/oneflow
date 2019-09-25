#include "oneflow/core/kernel/arc_face_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void ArcFaceForward(DeviceCtx* ctx, const Blob* in, const Blob* label, const int64_t lower_bound,
                    const T cos_m, const T sin_m, Blob* sin_theta_data, Blob* out) {
  ArcFaceKernelUtil<device_type, T, K>::Forward(
      ctx, in->shape().At(0), in->shape().At(1), in->dptr<T>(), label->dptr<K>(), lower_bound,
      cos_m, sin_m, sin_theta_data->mut_dptr<T>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void ArcFaceBackward(DeviceCtx* ctx, const Blob* out_diff, const int64_t lower_bound, const T cos_m,
                     const T sin_m, const Blob* label, const Blob* sin_theta_data, Blob* in_diff) {
  ArcFaceKernelUtil<device_type, T, K>::Backward(
      ctx, out_diff->shape().At(0), out_diff->shape().At(1), out_diff->dptr<T>(), label->dptr<K>(),
      lower_bound, cos_m, sin_m, sin_theta_data->dptr<T>(), in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct ArcFaceSwitchUtil final {
#define MAKE_ARC_FACE_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_ARC_FACE_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(ArcFaceForward);
  DEFINE_ARC_FACE_STATIC_SWITCH_FUNC(ArcFaceBackward);
#undef DEFINE_ARC_FACE_STATIC_SWITCH_FUNC
#undef MAKE_ARC_FACE_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& ArcFaceKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().arc_face_conf();
}

template<DeviceType device_type, typename T>
void ArcFaceKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  lower_bound_ = 0;
  if (parallel_ctx->policy() == kModelParallel) {
    auto& conf = this->op_conf().arc_face_conf();
    BalancedSplitter splitter(conf.depth(), parallel_ctx->parallel_num());
    lower_bound_ = splitter.At(parallel_ctx->parallel_id()).begin();
  }
}

template<DeviceType device_type, typename T>
void ArcFaceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float margin = this->op_conf().arc_face_conf().margin();
  const T cos_m = cos(margin);
  const T sin_m = sqrt(1 - cos_m * cos_m);
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
  ArcFaceSwitchUtil<device_type, T>::SwitchArcFaceForward(
      SwitchCase(BnInOp2Blob("label")->data_type()), ctx.device_ctx, BnInOp2Blob("in"),
      BnInOp2Blob("label"), lower_bound_, cos_m, sin_m, BnInOp2Blob("sin_theta_data"),
      BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void ArcFaceKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float margin = this->op_conf().arc_face_conf().margin();
  const T cos_m = cos(margin);
  const T sin_m = sqrt(1 - cos_m * cos_m);
  BnInOp2Blob("in_diff")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
  ArcFaceSwitchUtil<device_type, T>::SwitchArcFaceBackward(
      SwitchCase(BnInOp2Blob("label")->data_type()), ctx.device_ctx, BnInOp2Blob(GenDiffBn("out")),
      lower_bound_, cos_m, sin_m, BnInOp2Blob("label"), BnInOp2Blob("sin_theta_data"),
      BnInOp2Blob(GenDiffBn("in")));
}

template<typename T, typename K>
struct ArcFaceKernelUtil<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const T* in, const K* label, const int64_t lower_bound, const T cos_m,
                      const T sin_m, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const T* out_diff, const K* label, const int64_t lower_bound, const T cos_m,
                       const T sin_m, const T* sin_theta_data, T* in_diff);
};

template<typename T, typename K>
void ArcFaceKernelUtil<DeviceType::kCPU, T, K>::Forward(DeviceCtx* ctx, const int64_t batch_num,
                                                        const int64_t labels_num, const T* in,
                                                        const K* label, const int64_t lower_bound,
                                                        const T cos_m, const T sin_m,
                                                        T* sin_theta_data, T* out) {
  UNIMPLEMENTED();
}

template<typename T, typename K>
void ArcFaceKernelUtil<DeviceType::kCPU, T, K>::Backward(DeviceCtx* ctx, const int64_t batch_num,
                                                         const int64_t labels_num,
                                                         const T* out_diff, const K* label,
                                                         const int64_t lower_bound, const T cos_m,
                                                         const T sin_m, const T* sin_theta_data,
                                                         T* in_diff) {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kArcFaceConf, ArcFaceKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
