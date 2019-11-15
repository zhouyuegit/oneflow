#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"

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

  std::vector<std::unique_ptr<OneRecReader>> reader_;
  std::vector<std::unique_ptr<PersistentInStream>> in_stream_;
  std::unique_ptr<int64_t> counter_;
};

void DecodeOneRecKernel::VirtualKernelInit() {
  const int64_t num_reader_threads = this->op_conf().decode_onerec_conf().num_reader_threads();
  const DecodeOneRecKernelConf& conf = this->kernel_conf().decode_onerec_conf();
  BalancedSplitter bs(conf.file().size(), num_reader_threads);
  FOR_RANGE(int64_t, i, 0, num_reader_threads) {
    std::vector<std::string> data_paths(
        {conf.file().cbegin() + bs.At(i).begin(), conf.file().cbegin() + bs.At(i).end()});
    in_stream_.emplace_back(
        std::make_unique<PersistentInStream>(DataFS(), data_paths, true, false));
    reader_.emplace_back(std::make_unique<BufferedOneRecReader>(
        in_stream_.at(i).get(), GetMaxVal<int64_t>(), conf.device_batch_size(), 2));
  }
  counter_.reset(new int64_t(0));
}

void DecodeOneRecKernel::Forward(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t reader_idx = *counter_ % reader_.size();
  *counter_ = *counter_ + 1;
  const int64_t device_batch_size = this->kernel_conf().decode_onerec_conf().device_batch_size();
  std::vector<std::shared_ptr<OneRecExampleWrapper>> records;
  CHECK_EQ(reader_.at(reader_idx)->Read(device_batch_size, &records), device_batch_size);
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

    ThreadPool* thread_pool = Global<ThreadMgr>::Get()->compute_thread_pool();
    const int64_t part_num =
        std::min(static_cast<int64_t>(thread_pool->thread_num()), device_batch_size);
    BalancedSplitter bs(device_batch_size, part_num);
    BlockingCounter bc(part_num);
    FOR_RANGE(int64_t, tid, 0, part_num) {
      const Range range = bs.At(tid);
      thread_pool->AddWork([range, instance_size, &bc, &records, &field, &blob] {
        FOR_RANGE(int64_t, j, range.begin(), range.end()) {
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
            std::memcpy(blob->mut_dptr<int32_t>() + j * instance_size, values->data(),
                        instance_size * sizeof(int32_t));
          } else if (blob->data_type() == DataType::kFloat) {
            CHECK_EQ(tensor->data_type(), onerec::TensorData::TensorData_Float32List);
            const onerec::Float32List* list = tensor->data_as_Float32List();
            CHECK_NOTNULL(list);
            const flatbuffers::Vector<float>* values = list->values();
            CHECK_NOTNULL(values);
            CHECK_EQ(values->size(), instance_size);
            std::memcpy(blob->mut_dptr<float>() + j * instance_size, values->data(),
                        instance_size * sizeof(float));
          } else {
            UNIMPLEMENTED();
          }
        }
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  }
}

REGISTER_KERNEL(OperatorConf::kDecodeOnerecConf, DecodeOneRecKernel);

}  // namespace oneflow
