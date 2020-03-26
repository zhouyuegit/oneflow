#include "oneflow/customized/detection/data_load_actor.h"

namespace oneflow {

void DetectionDataLoadActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&DetectionDataLoadActor::HandlerNormal);
}

void DetectionDataLoadActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void DetectionDataLoadActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(kDetectionDataLoad, DetectionDataLoadActor);

}  // namespace oneflow
