#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"

namespace oneflow {

void CUPTIAPI kernelCallback(KernelTrace* kt_ptr, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cudaStream_t stream = nullptr;
    switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020: {
        stream = ((cudaConfigureCall_v3020_params*)(cbInfo->functionParams))->stream;
        break;
      }
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
        stream = ((cudaLaunchKernel_v7000_params*)(cbInfo->functionParams))->stream;
        break;
      }
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000: {
        stream = ((cudaLaunchKernel_ptsz_v7000_params*)(cbInfo->functionParams))->stream;
        break;
      }
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000: {
        stream = ((cudaLaunchCooperativeKernel_ptsz_v9000_params*)(cbInfo->functionParams))->stream;
        break;
      }
    }
    if (stream != nullptr) {
      std::unique_lock<std::mutex> lock(kt_ptr->count_mutex);
      kt_ptr->stream2launch_count[stream]++;
    }
  }
}

ThreadMgr::~ThreadMgr() {
  CUptiResult cuptierr;
  cuptierr = cuptiUnsubscribe(GetMutKernelTrace()->subscriber);
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

  if (Global<RuntimeCtx>::Get()->is_experiment_phase() == true) {
    kernel_trace_.reset(new KernelTrace());
    CUptiResult cuptierr;
    cuptierr = cuptiSubscribe(&GetMutKernelTrace()->subscriber, (CUpti_CallbackFunc)kernelCallback,
                              GetMutKernelTrace());
    CHECK_EQ(cuptierr, CUPTI_SUCCESS);
    cuptierr = cuptiEnableDomain(1, GetMutKernelTrace()->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    CHECK_EQ(cuptierr, CUPTI_SUCCESS);
  }
}
}  // namespace oneflow
