#ifndef ONEFLOW_CORE_PERSISTENCE_GZIP_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_GZIP_IN_STREAM_H_

#include "oneflow/core/persistence/in_stream.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include <zlib.h>

namespace oneflow {

class GZIPInStream : public InStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GZIPInStream);
  explicit GZIPInStream(std::unique_ptr<PersistentInStream>&& in_stream);
  ~GZIPInStream() override;

  int64_t Read(char* s, size_t n) override;

 private:
  std::unique_ptr<PersistentInStream> in_stream_;
  z_stream inflate_s_;
  std::vector<char> input_buf_;
  std::vector<char> output_buf_;
  int64_t output_buf_pos_ = 0;
  int64_t output_buf_size_ = 0;
  int64_t input_buf_pos_ = 0;
  int64_t input_buf_size_ = 0;
  bool is_eof_ = false;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_GZIP_IN_STREAM_H_
