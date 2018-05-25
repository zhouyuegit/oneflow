#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include <cupti.h>
#include <cuda.h>

namespace oneflow {

void CUPTIAPI kernelCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                             const CUpti_CallbackData* cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    auto id_it =
        Global<ThreadMgr>::Get()->linux_thread_id2thread_id.find(std::this_thread::get_id());
    CHECK(id_it != Global<ThreadMgr>::Get()->linux_thread_id2thread_id.end());
    int64_t actor_id = Global<ThreadMgr>::Get()->current_actor_id.at(id_it->second);
    Global<ThreadMgr>::Get()->kernel_launch_count.at(actor_id)++;
  }
}

ThreadMgr::~ThreadMgr() {
  CUptiResult cuptierr;
  CUpti_SubscriberHandle subscriber;
  cuptierr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)kernelCallback, NULL);
  CHECK_EQ(cuptierr, CUPTI_SUCCESS);
  cuptierr = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_EQ(cuptierr, CUPTI_SUCCESS);

  for (size_t i = 0; i < threads_.size(); ++i) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    threads_[i]->GetMsgChannelPtr()->Send(msg);
    delete threads_[i];
    LOG(INFO) << "actor thread " << i << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) { return threads_.at(thrd_id); }

ThreadMgr::ThreadMgr(const Plan& plan) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  int64_t thrd_id = 0;

  const OneMachineBufInfo& info = plan.buf_info().Get(Global<MachineCtx>::Get()->this_machine_id());

#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, 4) {
    FOR_RANGE(int64_t, dev_phy_id, 0, job_desc->GpuDeviceNum()) {
      threads_.push_back(new GpuThread(thrd_id, dev_phy_id, info.buf_size(thrd_id)));
      thrd_id += 1;
    }
  }
#endif
  FOR_RANGE(int64_t, i, 0, job_desc->CpuDeviceNum()) {
    threads_.push_back(new CpuThread(thrd_id, info.buf_size(thrd_id)));
    thrd_id += 1;
  }
  FOR_RANGE(int64_t, i, 0, job_desc->PersistenceWorkerNum()) {
    threads_.push_back(new CpuThread(thrd_id++, 0));
  }
  threads_.push_back(new CpuThread(thrd_id++, 0));  // comm_net

  int64_t th_id = 0;
  for (auto thread : threads_) {
    std::thread::id linux_thread_id = thread->mut_actor_thread().get_id();
    CHECK(linux_thread_id2thread_id.insert({linux_thread_id, th_id}).second);
    th_id++;
  }
}
}  // namespace oneflow
