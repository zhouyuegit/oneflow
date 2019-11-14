#ifndef ONEFLOW_CORE_RECORD_ONEREC_READER_H_
#define ONEFLOW_CORE_RECORD_ONEREC_READER_H_

#include <onerec_generated.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/common/buffer.h"

namespace oneflow {

class OneRecExampleWrapper {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneRecExampleWrapper);
  OneRecExampleWrapper(std::unique_ptr<char[]>&& data, int32_t size)
      : size_(size), data_(std::move(data)) {
    const auto buffer = reinterpret_cast<const uint8_t*>(data_.get());
    flatbuffers::Verifier verifier(buffer, static_cast<size_t>(size_));
    CHECK(onerec::VerifyExampleBuffer(verifier));
    example_ = onerec::GetExample(buffer);
  }
  ~OneRecExampleWrapper() = default;

  const onerec::Example* GetExample() { return example_; }

 private:
  int32_t size_;
  std::unique_ptr<char[]> data_;
  const onerec::Example* example_;
};

class OneRecReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneRecReader);
  OneRecReader() = default;
  virtual ~OneRecReader() = default;

  virtual size_t Read(size_t n, std::vector<std::shared_ptr<OneRecExampleWrapper>>* records) = 0;
};

class BufferedOneRecReader final : public OneRecReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BufferedOneRecReader);
  BufferedOneRecReader(PersistentInStream* in, size_t buffer_size)
      : BufferedOneRecReader(in, GetMaxVal<int64_t>(), buffer_size) {}
  BufferedOneRecReader(PersistentInStream* in, size_t num_max_read, size_t buffer_size);
  ~BufferedOneRecReader() override;

 private:
  size_t Read(size_t n, std::vector<std::shared_ptr<OneRecExampleWrapper>>* records) override;

  PersistentInStream* in_stream_;
  size_t num_read_;
  const size_t num_max_read_;
  const size_t buffer_size_;
  Buffer<std::shared_ptr<OneRecExampleWrapper>> chunk_buffer_;
  std::thread reader_thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_ONEREC_READER_H_
