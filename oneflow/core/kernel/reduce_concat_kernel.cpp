#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class ReduceConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatKernel);
  ReduceConcatKernel() = default;
  ~ReduceConcatKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void ReduceConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BatchMemcpyParams batch_memcpy_params{};
  batch_memcpy_params.num_params = 0;
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int, in_bn_id, 0, this->op_attribute().input_bns().size()) {
    char* dst_cur_dptr = out_blob->mut_dptr<char>()
                         + this->kernel_conf().reduce_concat_conf().data_offset().Get(in_bn_id);
    Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
    size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
    const char* in_dptr = in_blob->dptr<char>();
    if (device_type == DeviceType::kGPU && in_byte_size <= kBatchMemcpyMaxSize) {
      batch_memcpy_params.dst[batch_memcpy_params.num_params] = dst_cur_dptr;
      batch_memcpy_params.src[batch_memcpy_params.num_params] = in_dptr;
      batch_memcpy_params.size[batch_memcpy_params.num_params] = in_byte_size;
      batch_memcpy_params.num_params += 1;
    } else {
      Memcpy<device_type>(ctx.device_ctx, dst_cur_dptr, in_dptr, in_byte_size);
    }
    if (batch_memcpy_params.num_params == kBatchMemcpyMaxParam) {
      BatchMemcpyKernelUtil<device_type>::Copy(ctx.device_ctx, batch_memcpy_params);
      batch_memcpy_params.num_params = 0;
    }
  }
  if (batch_memcpy_params.num_params != 0) {
    BatchMemcpyKernelUtil<device_type>::Copy(ctx.device_ctx, batch_memcpy_params);
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceConcatConf, ReduceConcatKernel);

}  // namespace oneflow
