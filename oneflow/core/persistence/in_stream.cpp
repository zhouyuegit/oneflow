#include "oneflow/core/persistence/in_stream.h"

namespace oneflow {

int32_t InStream::ReadFully(char* s, size_t n) {
  int64_t read = Read(s, n);
  if (read <= 0) { return -1; }
  CHECK_EQ(read, n);
  return 0;
}

}  // namespace oneflow
