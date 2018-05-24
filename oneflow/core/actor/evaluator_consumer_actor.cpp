#include "oneflow/core/actor/evaluator_consumer_actor.h"

namespace oneflow {

void EvalConsumerActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&EvalConsumerActor::HandlerNormal);
}

REGISTER_ACTOR(TaskType::kEvalConsumer, EvalConsumerActor);

}  // namespace oneflow
