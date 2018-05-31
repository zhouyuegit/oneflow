#ifndef ONEFLOW_CORE_CONTROL_CTRL_CALL_H_
#define ONEFLOW_CORE_CONTROL_CTRL_CALL_H_

#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

class CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCallIf);
  virtual ~CtrlCallIf() = default;

  virtual void Process() = 0;
  virtual void SendResponse() = 0;

 protected:
  CtrlCallIf() = default;

 private:
};

template<CtrlMethod kMethod>
class CtrlCall final : public CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCall);
  CtrlCall() : status_(Status::kBeforeHandleRequest), responder_(&server_ctx_) {}
  ~CtrlCall() = default;

  using request_type = Reqeust<kMethod>;
  using response_type = Response<kMethod>;
  static constexpr const size_t value = (size_t)kMethod;
  const request_type& request() const { return request_; }

  request_type* mut_request() { return &request_; }
  response_type* mut_response() { return &response_; }
  grpc::ServerContext* mut_server_ctx() { return &server_ctx_; }
  grpc::ServerAsyncResponseWriter<response_type>* mut_responder() { return &responder_; }
  void set_request_handler(std::function<void()> val) { request_handler_ = std::move(val); }

  void Process() override {
    switch (status_) {
      case Status::kBeforeHandleRequest: {
        request_handler_();
        return;
      }
      case Status::kBeforeDelete: {
        delete this;
        return;
      }
    }
  }

  void SendResponse() override {
    responder_.Finish(response_, grpc::Status::OK, this);
    status_ = Status::kBeforeDelete;
  }

 private:
  enum class Status { kBeforeHandleRequest, kBeforeDelete };

  Status status_;
  request_type request_;
  response_type response_;
  grpc::ServerContext server_ctx_;
  grpc::ServerAsyncResponseWriter<response_type> responder_;
  std::function<void()> request_handler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CALL_H_
