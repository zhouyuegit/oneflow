#ifndef ONEFLOW_CORE_ACTOR_EVALUATOR_MDUPDT_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_EVALUATOR_MDUPDT_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class EvalMdUpdtActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EvalMdUpdtActor);
  EvalMdUpdtActor() = default;
  ~EvalMdUpdtActor() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_EVALUATOR_MDUPDT_ACTOR_H_
