#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_trace_desc.pb.h"

#include <cupti.h>
namespace oneflow {
class KernelTrace final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTrace);
  KernelTrace() = default;
  ~KernelTrace() = default;

  CUpti_SubscriberHandle subscriber;
  HashMap<cudaStream_t, int64_t> stream2launch_count;
  std::mutex count_mutex;
  KernelTraceDesc desc;
};
}  // namespace oneflow