#include "oneflow/core/persistence/gzip_in_stream.h"

namespace oneflow {

constexpr int32_t kGZIPInStreamDefaultInputBufferSize = 64 * 1024 * 1024;
constexpr int32_t kGZIPInStreamDefaultOutputBufferSize = 64 * 1024 * 1024;

GZIPInStream::GZIPInStream(std::unique_ptr<PersistentInStream>&& in_stream)
    : in_stream_(std::move(in_stream)) {
  inflate_s_.zalloc = Z_NULL;
  inflate_s_.zfree = Z_NULL;
  inflate_s_.opaque = Z_NULL;
  inflate_s_.avail_in = 0;
  inflate_s_.next_in = Z_NULL;
  constexpr int32_t window_bits = 15 + 32;
  CHECK_EQ(inflateInit2(&inflate_s_, window_bits), Z_OK);
  input_buf_.resize(kGZIPInStreamDefaultInputBufferSize);
  output_buf_.resize(kGZIPInStreamDefaultOutputBufferSize);
}

GZIPInStream::~GZIPInStream() { CHECK_EQ(inflateEnd(&inflate_s_), Z_OK); }

int64_t GZIPInStream::Read(char* s, size_t n) {
  if (is_eof_) { return -1; }
  int64_t read = 0;
  while (read < n) {
    const int64_t max_to_read = n - read;
    if (output_buf_pos_ < output_buf_size_) {
      const int32_t max_copy_size = std::min(max_to_read, output_buf_size_ - output_buf_pos_);
      std::memcpy(s + read, output_buf_.data() + output_buf_pos_, max_copy_size);
      read += max_copy_size;
      output_buf_pos_ += max_copy_size;
      if (output_buf_pos_ == output_buf_size_) {
        output_buf_pos_ = 0;
        output_buf_size_ = 0;
      }
      continue;
    }
    if (input_buf_pos_ == input_buf_size_) {
      int64_t read_size = in_stream_->Read(input_buf_.data(), input_buf_.size());
      if (read_size <= 0) {
        is_eof_ = true;
        break;
      } else {
        input_buf_pos_ = 0;
        input_buf_size_ = read_size;
      }
    }
    inflate_s_.next_in = reinterpret_cast<z_const Bytef*>(input_buf_.data() + input_buf_pos_);
    inflate_s_.avail_in = input_buf_size_ - input_buf_pos_;
    inflate_s_.next_out = reinterpret_cast<z_const Bytef*>(output_buf_.data());
    inflate_s_.avail_out = output_buf_.size();
    int32_t status = inflate(&inflate_s_, Z_FINISH);
    CHECK(status == Z_STREAM_END || status == Z_OK || status == Z_BUF_ERROR);
    output_buf_pos_ = 0;
    output_buf_size_ = output_buf_.size() - inflate_s_.avail_out;
    input_buf_pos_ = input_buf_size_ - inflate_s_.avail_in;
  }
  if (read == 0) {
    return -1;
  } else {
    return read;
  }
}

}  // namespace oneflow
