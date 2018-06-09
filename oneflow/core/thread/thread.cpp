#include "oneflow/core/thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  actor_thread_.join();
  CHECK(id2task_.empty());
  msg_channel_.CloseSendEnd();
  msg_channel_.CloseReceiveEnd();
}

void Thread::AddTask(const TaskProto& task) {
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  CHECK(id2task_.emplace(task.task_id(), task).second);
}

void Thread::PollMsgChannel(const ThreadCtx& thread_ctx) {
  ActorMsg msg;
  while (true) {
    CHECK_EQ(msg_channel_.Receive(&msg), 0);
    if (msg.msg_type() == ActorMsgType::kCmdMsg) {
      if (msg.actor_cmd() == ActorCmd::kStopThread) {
        CHECK(id2actor_ptr_.empty());
        break;
      } else if (msg.actor_cmd() == ActorCmd::kConstructActor) {
        ConstructActor(msg.dst_actor_id(), thread_ctx);
        continue;
      } else {
        // do nothing
      }
    }
    int64_t actor_id = msg.dst_actor_id();
    auto actor_it = id2actor_ptr_.find(actor_id);
    CHECK(actor_it != id2actor_ptr_.end());
    int process_msg_ret = actor_it->second->ProcessMsg(msg);
    if (process_msg_ret == 1) {
      LOG(INFO) << "thread " << thrd_id_ << " deconstruct actor " << actor_id;
      CollectTraceDesc(actor_it->second);
      id2actor_ptr_.erase(actor_it);
      Global<RuntimeCtx>::Get()->DecreaseCounter("running_actor_cnt");
    } else {
      CHECK_EQ(process_msg_ret, 0);
    }
  }
}

void Thread::ConstructActor(int64_t actor_id, const ThreadCtx& thread_ctx) {
  LOG(INFO) << "thread " << thrd_id_ << " construct actor " << actor_id;
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  auto task_it = id2task_.find(actor_id);
  CHECK(id2actor_ptr_.emplace(actor_id, NewActor(task_it->second, thread_ctx)).second);
  id2task_.erase(task_it);
  Global<RuntimeCtx>::Get()->DecreaseCounter("constructing_actor_cnt");
}

void Thread::CollectTraceDesc(const std::unique_ptr<Actor>& actor_ptr) {
  cudaStream_t cuda_stream = actor_ptr->GetCudaStream();
  if (cuda_stream == nullptr) { return; }
  auto kt_ptr = Global<RuntimeCtx>::Get()->GetMutKernelTrace();
  auto count_it = kt_ptr->stream2launch_count.find(actor_ptr->GetCudaStream());
  if (count_it != kt_ptr->stream2launch_count.end()) {
    KernelLaunchCount* cnt = new KernelLaunchCount();
    cnt->set_count(count_it->second);
    cnt->set_thread_id(thrd_id_);
    (*kt_ptr->desc.mutable_stream_id2count())[actor_ptr->GetLocalWorkStreamId()] = *cnt;
  }
}
}  // namespace oneflow
