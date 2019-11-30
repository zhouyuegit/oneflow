#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutNopOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutNopOpConTickInputHelper() : MutOpConTickInputHelper() {}
  ~MutNopOpConTickInputHelper() = default;

  bool VirtualIsTickInputBound() const override { return op_conf().nop_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_nop_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kNopConf, MutNopOpConTickInputHelper);

}  // namespace oneflow
