#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

CtrlService::Stub::Stub(std::shared_ptr<grpc::ChannelInterface> channel)
    : arr_(to_array(channel)), channel_(channel) {}

std::unique_ptr<CtrlService::Stub> CtrlService::NewStub(const std::string& addr) {
  return of_make_unique<Stub>(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
}

CtrlService::AsyncService::AsyncService() {
  for (int32_t i = 0; i < kCtrlMethodNum; ++i) {
    AddMethod(new grpc::RpcServiceMethod(GetMethodName(static_cast<CtrlMethod>(i)),
                                         grpc::RpcMethod::NORMAL_RPC, nullptr));
    grpc::Service::MarkMethodAsync(i);
  }
}

}  // namespace oneflow
