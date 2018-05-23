#include "oneflow/core/actor/evaluator_model_update_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void EvalMdUpdtActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  model_tmp_regst_desc_id_ = Name2SoleRegstDescId("model_tmp");
  init_remaining_cnt_ = 0;
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (model_tmp_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  OF_SET_MSG_HANDLER(&EvalMdUpdtActor::HandlerInitModelAndModelTmp);
}

void EvalMdUpdtActor::InitRegstBySendToFw(const int64_t regst_desc_id) {
  if (regst_desc_id == -1) { return; }
  Regst* regst = GetCurWriteableRegst(regst_desc_id);
  ActorMsg msg = ActorMsg::BuildRegstMsgToConsumer(actor_id(), related_init_model_actor_id_, regst);
  Global<ActorMsgBus>::Get()->SendMsg(msg);
}

int EvalMdUpdtActor::HandlerInitModelAndModelTmp(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitModel);
    InitRegstBySendToFw(model_regst_desc_id_);
    InitRegstBySendToFw(model_tmp_regst_desc_id_);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    init_remaining_cnt_ -= 1;
  } else {
    UNIMPLEMENTED();
  }
  if (init_remaining_cnt_ == 0) {
    OF_SET_MSG_HANDLER(&EvalMdUpdtActor::HandlerSendInitialModel);
    Global<RuntimeCtx>::Get()->DecreaseCounter("model_init_cnt");
  }
  return 0;
}

int EvalMdUpdtActor::HandlerSendInitialModel(const ActorMsg& msg) { return 0; }

REGISTER_ACTOR(TaskType::kEvalMdUpdt, EvalMdUpdtActor);

}  // namespace oneflow
