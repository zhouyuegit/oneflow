#include "oneflow/core/kernel/one_hot_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void OneHot(
    DeviceCtx* ctx, const Blob* indices, int64_t lower_bound, int64_t upper_bound, Blob* out) {
  OneHotKernelUtil<device_type, T, K>::Encode(
      ctx, indices->dptr<K>(), indices->shape().elem_cnt(), lower_bound, upper_bound,
      out->mut_dptr<T>());
}

}  // namespace

template<DeviceType device_type, typename T>
struct OneHotUtil final {
#define MAKE_ONE_HOT_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, OneHot, MAKE_ONE_HOT_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_ONE_HOT_SWITCH_ENTRY
};

template<DeviceType device_type, typename T>
const PbMessage& OneHotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().one_hot_conf();
}

template<DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  if (parallel_ctx->policy() == kModelParallel) {
    auto& conf = this->op_conf().one_hot_conf();
    BalancedSplitter splitter(conf.depth(), parallel_ctx->parallel_num());
    lower_bound_ = splitter.At(parallel_ctx->parallel_id()).begin();
    upper_bound_ = splitter.At(parallel_ctx->parallel_id()).end();
  } else if (parallel_ctx->policy() == kDataParallel) {
    auto& conf = this->op_conf().one_hot_conf();
    lower_bound_ = 0;
    upper_bound_ = conf.depth();
  } else {
    UNIMPLEMENTED();
  }
 }


template<DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  Blob* out = BnInOp2Blob("out");
  OneHotUtil<device_type, T>::SwitchOneHot(SwitchCase(indices->data_type()), ctx.device_ctx,
                                           indices, lower_bound_, upper_bound_, out);
}

template<typename T, typename K>
struct OneHotKernelUtil<DeviceType::kCPU, T, K> final {
  static void Encode(DeviceCtx* ctx, const K* indices, int64_t num_indices, int64_t lower_bound,
      int64_t upper_bound, T* out);
};

template<typename T, typename K>
void OneHotKernelUtil<DeviceType::kCPU, T, K>::Encode(DeviceCtx* ctx, const K* indices,
                                                      int64_t num_indices, int64_t lower_bound,
                                                      int64_t upper_bound, T* out) {
  const int64_t length = upper_bound - lower_bound;
  Memset<kCPU>(ctx, out, 0, num_indices * length * sizeof(T));
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const K idx = indices[i];
    CHECK_GE(idx, 0);
    K offset = idx % length;
    if (offset >= lower_bound && offset < upper_bound) {
      out[idx / length + offset] = OneVal<T>::value;
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kOneHotConf, OneHotKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
