#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/record/onerec_reader.h"

namespace oneflow {

class CtrBatchGeneratorKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrBatchGeneratorKernel);
  CtrBatchGeneratorKernel() = default;
  ~CtrBatchGeneratorKernel() override = default;

 private:
  void VirtualKernelInit() override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<OneRecReader> reader_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

void CtrBatchGeneratorKernel::VirtualKernelInit() {
  const CtrBatchGeneratorOpConf& conf = this->op_conf().ctr_batch_generator_conf();
  std::vector<std::string> files({conf.file().cbegin(), conf.file().cend()});
  in_stream_.reset(new PersistentInStream(DataFS(), files, true, false));
  reader_.reset(
      new BufferedOneRecReader(in_stream_.get(), GetMaxVal<int64_t>(), conf.batch_size(), 256));
}

void CtrBatchGeneratorKernel::Forward(const KernelCtx& ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const CtrBatchGeneratorOpConf& conf = this->op_conf().ctr_batch_generator_conf();
  std::vector<std::shared_ptr<OneRecExampleWrapper>> records;
  const int32_t batch_size = conf.batch_size();
  const int32_t num_partition = conf.num_partition();
  const int32_t num_slot = conf.num_slot();
  records.reserve(batch_size);
  Blob* label_blob = BnInOp2Blob("label");
  int8_t* label_ptr = label_blob->mut_dptr<int8_t>();
  std::vector<Blob*> feature_id_blob_vec;
  std::vector<Blob*> feature_slot_blob_vec;
  std::vector<int32_t*> feature_id_ptr_vec;
  std::vector<int32_t*> feature_slot_ptr_vec;
  std::vector<int32_t> partition_counter(num_partition, 0);
  int32_t* partition_counter_ptr = partition_counter.data();
  for (int32_t i = 0; i < num_partition; ++i) {
    Blob* feature_id_blob = BnInOp2Blob(GenRepeatedBn("feature_id", i));
    feature_id_blob_vec.push_back(feature_id_blob);
    feature_id_ptr_vec.push_back(feature_id_blob->mut_dptr<int32_t>());
    Blob* feature_slot_blob = BnInOp2Blob(GenRepeatedBn("feature_slot", i));
    feature_slot_blob_vec.push_back(feature_slot_blob);
    feature_slot_ptr_vec.push_back(feature_slot_blob->mut_dptr<int32_t>());
  }
  CHECK_EQ(reader_->Read(batch_size, &records), batch_size);

#pragma omp parallel for default(none) shared(records) shared(label_ptr) \
    shared(partition_counter_ptr) shared(feature_id_ptr_vec) shared(feature_slot_ptr_vec)
  for (int32_t i = 0; i < batch_size; ++i) {
    const onerec::Example* example = records.at(i)->GetExample();
    CHECK_NOTNULL(example);
    const onerec::Feature* label = example->features()->LookupByKey("label");
    CHECK_NOTNULL(label);
    const onerec::Tensor* label_tensor = label->tensor();
    CHECK_NOTNULL(label_tensor);
    CHECK_EQ(label_tensor->data_type(), onerec::TensorData_Int8List);
    const flatbuffers::Vector<int8_t>* label_values = label_tensor->data_as_Int8List()->values();
    CHECK_EQ(label_values->size(), 1);
    label_ptr[i] = label_values->Get(0);
    const onerec::Feature* feature_id = example->features()->LookupByKey("feature_id");
    CHECK_NOTNULL(feature_id);
    const onerec::Tensor* feature_id_tensor = feature_id->tensor();
    CHECK_NOTNULL(feature_id_tensor);
    CHECK_EQ(feature_id_tensor->data_type(), onerec::TensorData_Int32List);
    const flatbuffers::Vector<int32_t>* feature_id_values =
        feature_id_tensor->data_as_Int32List()->values();
    const onerec::Feature* feature_slot = example->features()->LookupByKey("feature_slot");
    CHECK_NOTNULL(feature_slot);
    const onerec::Tensor* feature_slot_tensor = feature_slot->tensor();
    CHECK_NOTNULL(feature_slot_tensor);
    CHECK_EQ(feature_slot_tensor->data_type(), onerec::TensorData_Int8List);
    const flatbuffers::Vector<int8_t>* feature_slot_values =
        feature_slot_tensor->data_as_Int8List()->values();
    const int32_t feature_length = feature_id_values->size();
    CHECK_EQ(feature_slot_values->size(), feature_length);
    int32_t slot_offset = i * num_slot;
    for (int32_t j = 0; j < feature_length; j++) {
      const int32_t id = feature_id_values->Get(j);
      const int32_t slot = feature_slot_values->Get(j);
      const int32_t part_id = id % num_partition;
      int32_t offset;
#pragma omp atomic capture
      {
        offset = partition_counter_ptr[part_id];
        partition_counter_ptr[part_id] += 1;
      }
      feature_id_ptr_vec.at(part_id)[offset] = id / num_partition;
      feature_slot_ptr_vec.at(part_id)[offset] = slot + slot_offset;
    }
  }

  for (int32_t i = 0; i < num_partition; ++i) {
    feature_id_blob_vec.at(i)->set_dim0_valid_num(0, partition_counter.at(i));
    feature_slot_blob_vec.at(i)->set_dim0_valid_num(0, partition_counter.at(i));
  }
}

REGISTER_KERNEL(OperatorConf::kCtrBatchGeneratorConf, CtrBatchGeneratorKernel);

}  // namespace oneflow
