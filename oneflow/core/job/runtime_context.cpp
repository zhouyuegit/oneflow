#include "oneflow/core/job/runtime_context.h"

namespace oneflow {
void CUPTIAPI kernelCallback(KernelTrace* kt_ptr, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cudaStream_t stream = nullptr;
    switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020: {
        stream = ((cudaEventRecord_v3020_params*)(cbInfo->functionParams))->stream;
        break;
      }
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

void RuntimeCtx::NewCounter(const std::string& name, int64_t val) {
  LOG(INFO) << "NewCounter " << name << " " << val;
  CHECK(counters_.emplace(name, of_make_unique<BlockingCounter>(val)).second);
}

void RuntimeCtx::DecreaseCounter(const std::string& name) {
  int64_t cur_val = counters_.at(name)->Decrease();
  LOG(INFO) << "DecreaseCounter " << name << ", current val is " << cur_val;
}

void RuntimeCtx::WaitUntilCntEqualZero(const std::string& name) {
  counters_.at(name)->WaitUntilCntEqualZero();
}

RuntimeCtx::RuntimeCtx(int64_t total_piece_num, bool is_experiment_phase) {
  total_piece_num_ = total_piece_num;
  is_experiment_phase_ = is_experiment_phase;
  if (is_experiment_phase) {
    kernel_trace_.reset(new KernelTrace());
    CUptiResult cuptierr;
    cuptierr = cuptiSubscribe(&GetMutKernelTrace()->subscriber, (CUpti_CallbackFunc)kernelCallback,
                              GetMutKernelTrace());
    CHECK_EQ(cuptierr, CUPTI_SUCCESS);
    cuptierr = cuptiEnableDomain(1, GetMutKernelTrace()->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    CHECK_EQ(cuptierr, CUPTI_SUCCESS);
  }
}

void RuntimeCtx::SaveTraceDesc(const std::string& path) {
  if (is_experiment_phase_ == false) { return; }
  if (GetMutKernelTrace()->subscriber != nullptr) {
    CUptiResult cuptierr;
    cuptierr = cuptiUnsubscribe(GetMutKernelTrace()->subscriber);
    CHECK_EQ(cuptierr, CUPTI_SUCCESS);
  }
  auto kt_ptr = Global<RuntimeCtx>::Get()->GetMutKernelTrace();
  PrintProtoToTextFile(kt_ptr->desc, path);
  // LOG(FATAL) << "thread_id2count_size " << kt_ptr->desc.thread_id2count_size();
}
}  // namespace oneflow
