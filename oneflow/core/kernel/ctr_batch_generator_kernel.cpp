#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

struct BatchData {
  std::vector<int8_t> label;
  std::vector<std::vector<int32_t>> feature_id;
  std::vector<std::vector<int32_t>> feature_slot;
};

class BatchGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGenerator);
  BatchGenerator(const std::vector<std::string>& files, int32_t batch_size, int32_t num_partition,
                 int32_t num_slot, int32_t max_num_feature)
      : batch_size_(batch_size),
        num_partition_(num_partition),
        num_slot_(num_slot),
        max_num_feature_(max_num_feature),
        buffer_(16) {
    BalancedSplitter bs(files.size(), 4);
    for (int32_t rid = 0; rid < 4; ++rid) {
      std::vector<std::string> reader_files;
      for (int32_t i = bs.At(rid).begin(); i < bs.At(rid).end(); ++i) {
        reader_files.push_back(files.at(i));
      }
      PersistentInStream* in_stream = new PersistentInStream(DataFS(), reader_files, true, false);
      in_streams_.emplace_back(in_stream);
      readers_.emplace_back(
          new BufferedOneRecReader(in_stream, GetMaxVal<int64_t>(), batch_size_, 64));
    }
    for (int32_t tid = 0; tid < 16; ++tid) {
      threads_.emplace_back(std::thread([this, tid]() {
        OneRecReader* reader = readers_.at(tid % 4).get();
        while (true) {
          std::vector<std::shared_ptr<OneRecExampleWrapper>> records;
          records.reserve(batch_size_);
          CHECK_EQ(reader->Read(batch_size_, &records), batch_size_);
          std::shared_ptr<BatchData> batch_data = std::make_shared<BatchData>();
          batch_data->label.reserve(batch_size_);
          batch_data->feature_id.resize(num_partition_);
          batch_data->feature_slot.resize(num_partition_);
          FOR_RANGE(int32_t, i, 0, num_partition_) {
            batch_data->feature_id.at(i).reserve(batch_size_ * max_num_feature_);
            batch_data->feature_slot.at(i).reserve(batch_size_ * max_num_feature_);
          }
          for (int32_t i = 0; i < batch_size_; ++i) {
            const onerec::Example* example = records.at(i)->GetExample();
            CHECK_NOTNULL(example);
            const onerec::Feature* label = example->features()->LookupByKey("label");
            CHECK_NOTNULL(label);
            const onerec::Tensor* label_tensor = label->tensor();
            CHECK_NOTNULL(label_tensor);
            CHECK_EQ(label_tensor->data_type(), onerec::TensorData_Int8List);
            const flatbuffers::Vector<int8_t>* label_values =
                label_tensor->data_as_Int8List()->values();
            CHECK_EQ(label_values->size(), 1);
            batch_data->label.push_back(label_values->Get(0));
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
            int32_t slot_offset = i * num_slot_;
            for (int32_t j = 0; j < feature_length; j++) {
              const int32_t id = feature_id_values->Get(j);
              const int32_t slot = feature_slot_values->Get(j);
              const int32_t part_id = id % num_partition_;
              batch_data->feature_id.at(part_id).push_back(id / num_partition_);
              batch_data->feature_slot.at(part_id).push_back(slot + slot_offset);
            }
          }
          const BufferStatus status = buffer_.Send(batch_data);
          if (status == BufferStatus::kBufferStatusErrorClosed) {
            break;
          } else {
            CHECK(status == BufferStatus::kBufferStatusSuccess);
          }
        }
      }));
    }
  }
  ~BatchGenerator() {
    buffer_.Close();
    for (std::thread& thread : threads_) { thread.join(); }
    for (auto& reader : readers_) { reader.reset(); };
    for (auto& in_stream : in_streams_) { in_stream.reset(); }
    // reader_.reset();
  }

  void FetchBatch(std::shared_ptr<BatchData>* batch) {
    BufferStatus status = buffer_.Receive(batch);
    CHECK_EQ(status, BufferStatus::kBufferStatusSuccess);
  }

 private:
  //  std::unique_ptr<OneRecReader> reader_;
  int32_t batch_size_;
  int32_t num_partition_;
  int32_t num_slot_;
  int32_t max_num_feature_;
  Buffer<std::shared_ptr<BatchData>> buffer_;
  std::vector<std::thread> threads_;
  std::vector<std::unique_ptr<OneRecReader>> readers_;
  std::vector<std::unique_ptr<PersistentInStream>> in_streams_;
};

}  // namespace

class CtrBatchGeneratorKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrBatchGeneratorKernel);
  CtrBatchGeneratorKernel() = default;
  ~CtrBatchGeneratorKernel() override;

 private:
  void VirtualKernelInit() override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<BatchGenerator> batch_generator_;
};

CtrBatchGeneratorKernel::~CtrBatchGeneratorKernel() { batch_generator_.reset(); }

void CtrBatchGeneratorKernel::VirtualKernelInit() {
  const CtrBatchGeneratorOpConf& conf = this->op_conf().ctr_batch_generator_conf();
  std::vector<std::string> files({conf.file().cbegin(), conf.file().cend()});
  batch_generator_.reset(new BatchGenerator(files, conf.batch_size(), conf.num_partition(),
                                            conf.num_slot(), conf.max_num_feature()));
}

void CtrBatchGeneratorKernel::Forward(const KernelCtx& ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::shared_ptr<BatchData> batch_data;
  batch_generator_->FetchBatch(&batch_data);
  const CtrBatchGeneratorOpConf& conf = this->op_conf().ctr_batch_generator_conf();
  Blob* label_blob = BnInOp2Blob("label");
  CHECK_EQ(label_blob->shape().elem_cnt(), batch_data->label.size());
  std::copy(batch_data->label.cbegin(), batch_data->label.cend(), label_blob->mut_dptr<int8_t>());
  for (int64_t i = 0; i < conf.num_partition(); ++i) {
    const std::vector<int32_t>& feature_id_vec = batch_data->feature_id.at(i);
    const std::vector<int32_t>& feature_slot_vec = batch_data->feature_slot.at(i);
    const int32_t num_feature = feature_id_vec.size();
    CHECK_EQ(feature_slot_vec.size(), num_feature);
    Blob* feature_id_blob = BnInOp2Blob(GenRepeatedBn("feature_id", i));
    CHECK_GE(feature_id_blob->static_shape().elem_cnt(), num_feature);
    std::copy(feature_id_vec.cbegin(), feature_id_vec.cend(), feature_id_blob->mut_dptr<int32_t>());
    feature_id_blob->set_dim0_valid_num(0, num_feature);
    Blob* feature_slot_blob = BnInOp2Blob(GenRepeatedBn("feature_slot", i));
    CHECK_GE(feature_slot_blob->static_shape().elem_cnt(), num_feature);
    std::copy(feature_slot_vec.cbegin(), feature_slot_vec.cend(),
              feature_slot_blob->mut_dptr<int32_t>());
    feature_slot_blob->set_dim0_valid_num(0, num_feature);
  }
}

REGISTER_KERNEL(OperatorConf::kCtrBatchGeneratorConf, CtrBatchGeneratorKernel);

}  // namespace oneflow
