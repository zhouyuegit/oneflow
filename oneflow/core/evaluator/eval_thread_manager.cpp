#include "oneflow/core/evaluator/eval_thread_manager.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"

namespace oneflow {

EvalThreadMgr::~EvalThreadMgr() {
  for (size_t i = 0; i < threads_.size(); ++i) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    threads_[i]->GetMsgChannelPtr()->Send(msg);
    delete threads_[i];
    LOG(INFO) << "Evaluator actor thread " << i << " finish";
  }
}

Thread* EvalThreadMgr::GetThrd(int64_t thrd_id) { return threads_.at(thrd_id); }

EvalThreadMgr::EvalThreadMgr() {
  int64_t thrd_id = 0;
#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, 1) { threads_.push_back(new GpuThread(thrd_id++, i)); }
#endif
}

}  // namespace oneflow
