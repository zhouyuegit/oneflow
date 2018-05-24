#ifndef ONEFLOW_CORE_ACTOR_EVALUATOR_CONSUMER_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_EVALUATOR_CONSUMER_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class EvalConsumerActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EvalConsumerActor);
  EvalConsumerActor() = default;
  ~EvalConsumerActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override{};
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {true, {}};
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_EVALUATOR_CONSUMER_ACTOR_H_
