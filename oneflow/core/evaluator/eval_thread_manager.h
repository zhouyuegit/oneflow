#ifndef ONEFLOW_CORE_EVALUATOR_EVAL_THREAD_MANAGER_H_
#define ONEFLOW_CORE_EVALUATOR_EVAL_THREAD_MANAGER_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/thread/thread.h"

namespace oneflow {

class EvalThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EvalThreadMgr);
  ~EvalThreadMgr();

  Thread* GetThrd(int64_t thrd_id);

 private:
  friend class Global<EvalThreadMgr>;
  EvalThreadMgr();

  std::vector<Thread*> threads_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EVALUATOR_EVAL_THREAD_MANAGER_H_
