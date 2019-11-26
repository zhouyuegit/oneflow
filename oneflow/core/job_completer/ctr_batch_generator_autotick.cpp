#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutCtrBatchGeneratorOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutCtrBatchGeneratorOpConTickInputHelper() : MutOpConTickInputHelper() {}
  ~MutCtrBatchGeneratorOpConTickInputHelper() = default;

  bool VirtualIsTickInputBound() const override {
    return op_conf().ctr_batch_generator_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_ctr_batch_generator_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kCtrBatchGeneratorConf, MutCtrBatchGeneratorOpConTickInputHelper);

}  // namespace oneflow
