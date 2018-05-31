#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_

#include <array>
#include <grpc++/alarm.h>
#include <grpc++/server_builder.h>
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/control/ctrl_call.h"

namespace oneflow {

namespace {
template<size_t... Idx>
static std::tuple<std::function<void(CtrlCall<(CtrlMethod)Idx>*)>...> ToTupleImpl(
    oneflow::index_sequence<Idx...>) {
  return {};
}

static auto ToTuple() -> decltype(ToTupleImpl(oneflow::make_index_sequence<kCtrlMethodNum>{})) {
  return {};
}
}  // namespace

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  CtrlServer() = delete;
  ~CtrlServer();

  CtrlServer(const std::string& server_addr);

 private:
  void HandleRpcs();
  void Init();

  void EnqueueRequests() {
    for_each_i(tp_, helper{this}, oneflow::make_index_sequence<kCtrlMethodNum>{});
  }

  template<CtrlMethod kMethod>
  void EnqueueRequest() {
    constexpr const size_t I = (size_t)kMethod;
    auto handler = std::get<I>(tp_);
    auto call = new CtrlCall<(CtrlMethod)I>();
    call->set_request_handler(std::bind(handler, call));
    grpc_service_->RequestAsyncUnary(I, call->mut_server_ctx(), call->mut_request(),
                                     call->mut_responder(), cq_.get(), cq_.get(), call);
  }

  template<CtrlMethod... kMethod>
  using Tuple = std::tuple<std::function<void(CtrlCall<kMethod>*)>...>;

  template<size_t... Idx, typename... Args>
  Tuple<(CtrlMethod)Idx...> make_tp(oneflow::index_sequence<Idx...>, Args... args) {
    return std::make_tuple(std::move(args)...);
  }

  template<typename F>
  void Add(F f) {
    using tuple_type = typename function_traits<F>::tuple_type;
    using arg_type =
        typename std::remove_pointer<typename std::tuple_element<0, tuple_type>::type>::type;

    std::get<arg_type::value>(tp_) = std::move(f);
  }

  struct helper {
    helper(CtrlServer* s) : s_(s) {}
    template<typename T, typename V>
    void operator()(const T& t, V) {
      s_->EnqueueRequest<(CtrlMethod)V::value>();
    }

    CtrlServer* s_;
  };

  decltype(ToTuple()) tp_;
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
