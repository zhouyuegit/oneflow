#include "oneflow/core/actor/evaluator_model_update_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace {

void RandInitBlob(Blob* blob) {
  // TODO, only support float and GPU here
  RandomGenerator rng(GetCurTime());
  rng.Uniform<kGPU, float>(blob->shape().elem_cnt(), blob->mut_dptr<float>());
}

}  // namespace

void EvalMdUpdtActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  model_tmp_regst_desc_id_ = Name2SoleRegstDescId("model_tmp");
  is_eof_ = false;
  init_remaining_cnt_ = 0;
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (model_tmp_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  if (related_init_model_actor_id_ == -1) {
    // this model update actor has no forward compute actor to init model.
    // So directly init the produced model regst by itself in the ActorInit.
    for (const auto& pair : task_proto.produced_regst_desc()) {
      Regst* regst = GetCurWriteableRegst(pair.second.regst_desc_id());
      for (const auto& pair : regst->lbi2blob()) {
        RandInitBlob(static_cast<Blob*>(pair.second.get()));
      }
    }
  }
  OF_SET_MSG_HANDLER(&EvalMdUpdtActor::HandlerInitModelAndModelTmp);
}

void EvalMdUpdtActor::InitRegstBySendToFw(const int64_t regst_desc_id) {
  if (regst_desc_id == -1) { return; }
  Regst* regst = GetCurWriteableRegst(regst_desc_id);
  CHECK_EQ(regst->regst_desc()->register_num(), 1);
  ActorMsg msg = ActorMsg::BuildRegstMsgToConsumer(actor_id(), related_init_model_actor_id_, regst);
  Global<ActorMsgBus>::Get()->SendMsg(msg);
}

int EvalMdUpdtActor::HandlerInitModelAndModelTmp(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitModel);
    if (related_init_model_actor_id_ == -1) {
      // this model update actor has no forward compute actor to init model.
      // So directly init the produced model regst by itself in the ActorInit.
      init_remaining_cnt_ = 0;
    } else {
      InitRegstBySendToFw(model_regst_desc_id_);
      InitRegstBySendToFw(model_tmp_regst_desc_id_);
    }
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

int EvalMdUpdtActor::HandlerSendInitialModel(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kSendInitialModel);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(0);
    return true;
  });
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (model_tmp_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  OF_SET_MSG_HANDLER(&EvalMdUpdtActor::HandlerWaitToEnd);
  return 0;
}

int EvalMdUpdtActor::HandlerWaitToEnd(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kRegstMsg) { init_remaining_cnt_ -= 1; }
  if (init_remaining_cnt_ != 0) {
    return 0;
  } else {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(nullptr);
    return 1;
  }
}

REGISTER_ACTOR(TaskType::kEvalMdUpdt, EvalMdUpdtActor);

}  // namespace oneflow
