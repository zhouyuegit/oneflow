#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_ACTOR_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class DetectionDataLoadActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DetectionDataLoadActor);
  DetectionDataLoadActor() = default;
  ~DetectionDataLoadActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_LOAD_ACTOR_H_
