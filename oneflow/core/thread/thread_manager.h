#ifndef ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/thread/thread.h"
#include <cupti.h>

namespace oneflow {

class KernelTrace final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTrace);
  KernelTrace(const int64_t threads_cnt) { current_actor_id.assign(threads_cnt, 0); };
  ~KernelTrace() = default;
  CUpti_SubscriberHandle subscriber;
  HashMap<std::thread::id, int64_t> linux_thread_id2thread_id;
  std::vector<int64_t> current_actor_id;  // size([actor_id, actor_id, actor_id]) = size_of_thread
  HashMap<int64_t, int64_t> actor_id2launch_count;
};

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ThreadMgr() = delete;
  ~ThreadMgr();

  KernelTrace* GetMutKernelTrace() const { return kernel_trace_.get(); }
  Thread* GetThrd(int64_t thrd_id);

 private:
  friend class Global<ThreadMgr>;
  ThreadMgr(const Plan& plan);

  std::unique_ptr<KernelTrace> kernel_trace_;
  std::vector<Thread*> threads_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
