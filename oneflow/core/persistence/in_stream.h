#ifndef ONEFLOW_CORE_PERSISTENCE_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_IN_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class InStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InStream);
  InStream() = default;
  virtual ~InStream() = default;

  virtual int32_t ReadFully(char* s, size_t n);
  virtual int64_t Read(char* s, size_t n) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_IN_STREAM_H_
