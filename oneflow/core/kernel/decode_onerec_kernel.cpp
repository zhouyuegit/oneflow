#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/record/onerec_reader.h"

namespace oneflow {

class DecodeOneRecKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOneRecKernel);
  DecodeOneRecKernel() = default;
  ~DecodeOneRecKernel() override = default;

 private:
  void VirtualKernelInit() override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<OneRecReader> reader_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

void DecodeOneRecKernel::VirtualKernelInit() {
  const DecodeOneRecKernelConf& conf = this->kernel_conf().decode_onerec_conf();
  std::vector<std::string> data_paths({conf.file().cbegin(), conf.file().cend()});
  in_stream_.reset(new PersistentInStream(DataFS(), data_paths, true, false));
  reader_.reset(new BufferedOneRecReader(in_stream_.get(), GetMaxVal<int64_t>(),
                                         conf.device_batch_size() * 8));
}

void DecodeOneRecKernel::Forward(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<std::shared_ptr<OneRecExampleWrapper>> records;
  const int64_t device_batch_size = this->kernel_conf().decode_onerec_conf().device_batch_size();
  CHECK_EQ(reader_->Read(device_batch_size, &records), device_batch_size);
  for (const std::string& bn : this->op_attribute().output_bns()) {
    Blob* blob = BnInOp2Blob(bn);
    Memset<DeviceType::kCPU>(ctx.device_ctx, blob->mut_dptr(), 0,
                             blob->ByteSizeOfDataContentField());
  }
}

REGISTER_KERNEL(OperatorConf::kDecodeOnerecConf, DecodeOneRecKernel);

}  // namespace oneflow
