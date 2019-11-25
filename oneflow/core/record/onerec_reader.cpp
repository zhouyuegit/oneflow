#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/blocking_counter.h"
#include <onerec_generated.h>

namespace oneflow {

constexpr int64_t MAX_CHUNK_SIZE = 64 * 1024 * 1024;  // 64M

namespace {

bool ReadChunk(PersistentInStream* is, size_t batch_size,
               std::vector<std::shared_ptr<OneRecExampleWrapper>>* batch) {
  for (size_t i = 0; i < batch_size; ++i) {
    int64_t magic;
    int32_t size;
    int32_t header_crc32;
    if (is->ReadFully(reinterpret_cast<char*>(&magic), sizeof(int64_t)) != 0) { return false; }
    CHECK_EQ(is->ReadFully(reinterpret_cast<char*>(&size), sizeof(int32_t)), 0);
    CHECK_EQ(is->ReadFully(reinterpret_cast<char*>(&header_crc32), sizeof(int32_t)), 0);
    CHECK_GE(size, 0);
    CHECK_LE(size, MAX_CHUNK_SIZE);
    const int32_t padded_size = RoundUp(size + 4, 8);
    std::unique_ptr<char[]> data;
    data.reset(new char[padded_size]);
    CHECK_EQ(is->ReadFully(data.get(), padded_size), 0);
    batch->emplace_back(new OneRecExampleWrapper(std::move(data), size));
  }
  return true;
}

}  // namespace

BufferedOneRecReader::BufferedOneRecReader(PersistentInStream* in, size_t num_max_read,
                                           size_t batch_size, size_t buffer_size)
    : in_stream_(in),
      num_read_(0),
      num_max_read_(num_max_read),
      batch_size_(batch_size),
      buffer_size_(buffer_size),
      chunk_buffer_(buffer_size) {
  reader_thread_ = std::thread([&]() {
    FOR_RANGE(int64_t, i, 0, num_max_read_) {
      std::vector<std::shared_ptr<OneRecExampleWrapper>> batch;
      if (ReadChunk(in_stream_, batch_size_, &batch)) {
        const BufferStatus status = chunk_buffer_.Send(batch);
        if (status == BufferStatus::kBufferStatusErrorClosed) {
          break;
        } else {
          CHECK(status == BufferStatus::kBufferStatusSuccess);
        }
      } else {
        chunk_buffer_.Close();
      }
    }
  });
}

BufferedOneRecReader::~BufferedOneRecReader() {
  chunk_buffer_.Close();
  reader_thread_.join();
}

size_t BufferedOneRecReader::Read(size_t n,
                                  std::vector<std::shared_ptr<OneRecExampleWrapper>>* records) {
  CHECK_EQ(n, batch_size_);
  BufferStatus status = chunk_buffer_.Receive(records);
  CHECK_EQ(status, BufferStatus::kBufferStatusSuccess);
  return n;
  const size_t can_read = std::min(n, num_max_read_ - num_read_);
  size_t cur_read = 0;
  FOR_RANGE(size_t, i, 0, can_read) { std::shared_ptr<OneRecExampleWrapper> chunk; }
  if (cur_read == 0) { return 0; }
  num_read_ += cur_read;
  return cur_read;
}

}  // namespace oneflow
