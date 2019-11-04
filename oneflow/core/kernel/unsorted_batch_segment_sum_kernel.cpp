#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/unsorted_batch_segment_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class UnsortedBatchSegmentSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnsortedBatchSegmentSumKernel);
  UnsortedBatchSegmentSumKernel() = default;
  ~UnsortedBatchSegmentSumKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
const PbMessage& UnsortedBatchSegmentSumKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().unsorted_batch_segment_sum_conf();
}

template<DeviceType device_type, typename T, typename K>
void UnsortedBatchSegmentSumKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("segment_ids");
  const Blob* data = BnInOp2Blob("data");
  Blob* out = BnInOp2Blob("out");
  const int64_t num_batch_axes = indices->shape().NumAxes() - 1;
  CHECK_GT(num_batch_axes, 0);
  const int64_t num_batches = indices->shape().Count(0, num_batch_axes);
  const int64_t num_indices = indices->shape().Count(num_batch_axes);
  const int64_t num_segments = this->op_conf().unsorted_batch_segment_sum_conf().num_segments();
  const int64_t instance_size = data->shape().Count(indices->shape().NumAxes());
  CHECK_EQ(out->shape().elem_cnt(), num_batches * num_segments * instance_size);
  UnsortedBatchSegmentSumKernelUtil<device_type, T, K>::UnsortedBatchSegmentSum(
      ctx.device_ctx, num_batches, num_indices, num_segments, instance_size, indices->dptr<K>(),
      data->dptr<T>(), out->mut_dptr<T>());
}

#define MAKE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_ENTRY(device_type_v, data_type_pair,         \
                                                     indices_type_pair)                     \
  NEW_REGISTER_KERNEL(                                                                      \
      OperatorConf::kUnsortedBatchSegmentSumConf,                                           \
      UnsortedBatchSegmentSumKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),        \
                                    OF_PP_PAIR_FIRST(indices_type_pair)>)                   \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                         \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)       \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())         \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                    \
                    == kernel_conf.unsorted_batch_segment_sum_conf().indices_data_type())); \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_UNSORTED_BATCH_SEGMENT_SUM_KERNEL_ENTRY

}  // namespace oneflow
