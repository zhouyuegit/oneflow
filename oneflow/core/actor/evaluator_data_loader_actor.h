#ifndef ONEFLOW_CORE_ACTOR_EVALUATOR_DATA_LOADER_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_EVALUATOR_DATA_LOADER_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class EvalDataLdActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EvalDataLdActor);
  EvalDataLdActor() = default;
  ~EvalDataLdActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override { return !IsCustomizedReadReady(); }

  bool is_eof_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_EVALUATOR_DATA_LOADER_ACTOR_H_
