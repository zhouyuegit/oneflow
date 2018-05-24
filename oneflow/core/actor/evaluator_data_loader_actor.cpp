#include "oneflow/core/actor/evaluator_data_loader_actor.h"

namespace oneflow {

void EvalDataLdActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_eof_ = true;
  OF_SET_MSG_HANDLER(&EvalDataLdActor::HandlerNormal);
}

void EvalDataLdActor::Act() {}

bool EvalDataLdActor::IsCustomizedReadReady() { return !is_eof_; }

REGISTER_ACTOR(TaskType::kEvalDataLd, EvalDataLdActor);

}  // namespace oneflow
