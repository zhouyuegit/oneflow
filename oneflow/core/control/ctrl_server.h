#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_

#include <array>
#include <grpc++/alarm.h>
#include <grpc++/server_builder.h>
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/control/ctrl_call.h"

namespace oneflow {

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  CtrlServer() = delete;
  ~CtrlServer();

  CtrlServer(const std::string& server_addr);

 private:
  void HandleRpcs();

  template<typename... Args>
  void init(Args... args) {
    static_assert(sizeof...(Args) == kCtrlMethodNum, "must equal");
    arr_ = {reinterpret_cast<Member>(args)...};
  }

  typedef void (CtrlServer::*Member)(void*);
  std::array<Member, kCtrlMethodNum> arr_;

  template<std::size_t I = 0, typename T>
  typename std::enable_if<I == array_size<T>::size>::type EnqueueRequests(T& arr){};

  template<std::size_t I = 0, typename T>
      typename std::enable_if < I<array_size<T>::size>::type EnqueueRequests(T& arr) {
    EnqueueRequest<I>();
    EnqueueRequests<I + 1>(arr);
  }

  template<size_t I>
  void EnqueueRequest();

  template<CtrlMethod I>
  void EnqueueRequest() {
    EnqueueRequest<(size_t)I>();
  }

  void LoadServerHandler(CtrlCall<CtrlMethod::kLoadServer>* call);
  void BarrierHandler(CtrlCall<CtrlMethod::kBarrier>* call);
  void TryLockHandler(CtrlCall<CtrlMethod::kTryLock>* call);
  void NotifyDoneHandler(CtrlCall<CtrlMethod::kNotifyDone>* call);
  void WaitUntilDoneHandler(CtrlCall<CtrlMethod::kWaitUntilDone>* call);
  void PushKVHandler(CtrlCall<CtrlMethod::kPushKV>* call);
  void ClearKVHandler(CtrlCall<CtrlMethod::kClearKV>* call);
  void PullKVHandler(CtrlCall<CtrlMethod::kPullKV>* call);
  void PushActEventHandler(CtrlCall<CtrlMethod::kPushActEvent>* call);
  void ClearHandler(CtrlCall<CtrlMethod::kClear>* call);
  void IncreaseCountHandler(CtrlCall<CtrlMethod::kIncreaseCount>* call);
  void EraseCountHandler(CtrlCall<CtrlMethod::kEraseCount>* call);
  void PushAvgActIntervalHandler(CtrlCall<CtrlMethod::kPushAvgActInterval>* call);

  std::unique_ptr<CtrlService::AsyncService> grpc_service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::thread loop_thread_;
  // Barrier
  HashMap<std::string, std::pair<std::list<CtrlCallIf*>, int32_t>> barrier_calls_;
  // TryLock, NotifyDone, WaitUntilDone
  HashMap<std::string, void*> name2lock_status_;
  // PushKV, ClearKV, PullKV
  HashMap<std::string, std::string> kv_;
  HashMap<std::string, std::list<CtrlCall<CtrlMethod::kPullKV>*>> pending_kv_calls_;
  // IncreaseCount, EraseCount
  HashMap<std::string, int32_t> count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
