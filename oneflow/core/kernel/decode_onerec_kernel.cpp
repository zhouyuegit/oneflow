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

  const PbRpf<DecodeOneRecFieldConf>& fields = this->op_conf().decode_onerec_conf().field();
  const int64_t field_size = this->op_attribute().output_bns().size();
  CHECK_EQ(fields.size(), field_size);
  FOR_RANGE(int64_t, i, 0, field_size) {
    const DecodeOneRecFieldConf& field = fields.Get(i);
    const std::string& bn = this->op_attribute().output_bns().Get(i);
    Blob* blob = BnInOp2Blob(bn);
    const Shape& blob_shape = blob->shape();
    CHECK_EQ(blob_shape.At(0), device_batch_size);
    const int64_t instance_size = blob_shape.Count(1);
    FOR_RANGE(int64_t, j, 0, device_batch_size) {
      CHECK_NOTNULL(records.at(j)->GetExample()->features());
      const onerec::Feature* feature =
          records.at(j)->GetExample()->features()->LookupByKey(field.key().c_str());
      CHECK_NOTNULL(feature);
      const onerec::Tensor* tensor = feature->tensor();
      CHECK_NOTNULL(tensor);
      if (blob->data_type() == DataType::kInt32) {
        CHECK_EQ(tensor->data_type(), onerec::TensorData::TensorData_Int32List);
        const onerec::Int32List* list = tensor->data_as_Int32List();
        CHECK_NOTNULL(list);
        const flatbuffers::Vector<int32_t>* values = list->values();
        CHECK_NOTNULL(values);
        CHECK_EQ(values->size(), instance_size);
        std::copy(values->cbegin(), values->cend(), blob->mut_dptr<int32_t>() + j * instance_size);
      } else if (blob->data_type() == DataType::kFloat) {
        CHECK_EQ(tensor->data_type(), onerec::TensorData::TensorData_Float32List);
        const onerec::Float32List* list = tensor->data_as_Float32List();
        CHECK_NOTNULL(list);
        const flatbuffers::Vector<float>* values = list->values();
        CHECK_NOTNULL(values);
        CHECK_EQ(values->size(), instance_size);
        std::copy(values->cbegin(), values->cend(), blob->mut_dptr<float>() + j * instance_size);
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

REGISTER_KERNEL(OperatorConf::kDecodeOnerecConf, DecodeOneRecKernel);

}  // namespace oneflow
