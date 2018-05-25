#ifndef ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/thread/thread.h"

namespace oneflow {

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ThreadMgr() = delete;
  ~ThreadMgr();

  Thread* GetThrd(int64_t thrd_id);
  HashMap<std::thread::id, int64_t> linux_thread_id2thread_id;
  std::vector<int64_t> current_actor_id;  // size([actor_id, actor_id, actor_id]) = size_of_thread
  std::vector<int64_t> kernel_launch_count;  // size([count, count, count]) = actor_id
 private:
  friend class Global<ThreadMgr>;
  ThreadMgr(const Plan& plan);

  std::vector<Thread*> threads_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
