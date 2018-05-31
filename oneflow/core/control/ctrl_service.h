#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_

#include <grpc++/grpc++.h>
#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/control/control.pb.h"

namespace oneflow {

#define CTRL_METHOD_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(LoadServer)    \
  OF_PP_MAKE_TUPLE_SEQ(Barrier)       \
  OF_PP_MAKE_TUPLE_SEQ(TryLock)       \
  OF_PP_MAKE_TUPLE_SEQ(NotifyDone)    \
  OF_PP_MAKE_TUPLE_SEQ(WaitUntilDone) \
  OF_PP_MAKE_TUPLE_SEQ(PushKV)        \
  OF_PP_MAKE_TUPLE_SEQ(ClearKV)       \
  OF_PP_MAKE_TUPLE_SEQ(PullKV)        \
  OF_PP_MAKE_TUPLE_SEQ(PushActEvent)  \
  OF_PP_MAKE_TUPLE_SEQ(Clear)         \
  OF_PP_MAKE_TUPLE_SEQ(IncreaseCount) \
  OF_PP_MAKE_TUPLE_SEQ(EraseCount)    \
  OF_PP_MAKE_TUPLE_SEQ(PushAvgActInterval)

#define ConactRequest(method) method##Request,
#define ConactReqponse(method) method##Response,
#define ConactEnum(method) k##method,
#define ConactStr(method) #method,

#define MAKE_META_DATA(...)                                                                  \
  enum class CtrlMethod { OF_PP_FOR_EACH_TUPLE(ConactEnum, CTRL_METHOD_SEQ) kCount };        \
  static const char* g_method_name[] = {OF_PP_FOR_EACH_TUPLE(ConactStr, CTRL_METHOD_SEQ)};   \
  using RequestType = std::tuple<OF_PP_FOR_EACH_TUPLE(ConactRequest, CTRL_METHOD_SEQ) bool>; \
  using ResponseType = std::tuple<OF_PP_FOR_EACH_TUPLE(ConactReqponse, CTRL_METHOD_SEQ) bool>;

MAKE_META_DATA()

constexpr const size_t kCtrlMethodNum = (size_t)CtrlMethod::kCount;
namespace {
const char* GetMethodName(CtrlMethod method) { return g_method_name[static_cast<int32_t>(method)]; }
const char* GetMethodName(size_t index) { return g_method_name[index]; }

}  // namespace

template<CtrlMethod kMethod>
using Reqeust = typename std::tuple_element<(size_t)kMethod, RequestType>::type;

template<CtrlMethod kMethod>
using Response = typename std::tuple_element<(size_t)kMethod, ResponseType>::type;

class CtrlService final {
 public:
  class Stub final {
   public:
    Stub(std::shared_ptr<grpc::ChannelInterface> channel);

    template<CtrlMethod kMethod>
    grpc::Status Method(grpc::ClientContext* context, const Reqeust<kMethod>& request,
                        Response<kMethod>* response) {
      size_t index = (size_t)kMethod;
      return grpc::BlockingUnaryCall(channel_.get(), arr_[index], context, request, response);
    }

   private:
    template<size_t I>
    grpc::RpcMethod get(const std::shared_ptr<grpc::ChannelInterface>& channel) {
      return grpc::RpcMethod({GetMethodName(I), grpc::RpcMethod::NORMAL_RPC, channel});
    }

    template<size_t... Indices>
    std::array<const grpc::RpcMethod, kCtrlMethodNum> to_array(
        oneflow::index_sequence<Indices...>,
        const std::shared_ptr<grpc::ChannelInterface>& channel) {
      return {{get<Indices>(channel)...}};
    }

    std::array<const grpc::RpcMethod, kCtrlMethodNum> to_array(
        const std::shared_ptr<grpc::ChannelInterface>& channel) {
      return to_array(oneflow::make_index_sequence<kCtrlMethodNum>{}, channel);
    }

    std::array<const grpc::RpcMethod, kCtrlMethodNum> arr_;
    std::shared_ptr<grpc::ChannelInterface> channel_;
  };

  static std::unique_ptr<Stub> NewStub(const std::string& addr);

  class AsyncService final : public grpc::Service {
   public:
    AsyncService();
    ~AsyncService() = default;
    using grpc::Service::RequestAsyncUnary;
  };
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_
