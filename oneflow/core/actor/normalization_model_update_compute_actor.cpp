#include "oneflow/core/actor/normalization_model_update_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void NormalizationMdUpdtCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("other_model");
  related_save_model_actor_id_ = task_proto.related_save_model_task_id();
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  model_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&NormalizationMdUpdtCompActor::HandlerInitModel);
}

void NormalizationMdUpdtCompActor::InitRegstBySendToFw(int64_t regst_desc_id) {
  if (regst_desc_id == -1) { return; }
  Regst* regst = GetCurWriteableRegst(regst_desc_id);
  ActorMsg msg = ActorMsg::BuildRegstMsgToConsumer(
      actor_id(), related_init_model_actor_id_, regst);
  ActorMsgBus::Singleton()->SendMsg(msg);
}

int NormalizationMdUpdtCompActor::HandlerInitModel(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitModel);
    InitRegstBySendToFw(model_regst_desc_id_);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    OF_SET_MSG_HANDLER(&NormalizationMdUpdtCompActor::HandlerSendInitialModel);
    RuntimeCtx::Singleton()->DecreaseCounter("model_init_cnt");
  } else {
    UNIMPLEMENTED();
  }
  return 0;
}

int NormalizationMdUpdtCompActor::HandlerSendInitialModel(
    const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  model_regst_ = GetCurWriteableRegst(model_regst_desc_id_);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(0);
    return true;
  });
  DecreaseRemainingEordCnt();
  AsyncSendEORDMsgForAllProducedRegstDesc();
  OF_SET_MSG_HANDLER(&NormalizationMdUpdtCompActor::HandlerNormal);
  return 0;
}

int NormalizationMdUpdtCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = actor_msg.regst();
    CHECK(regst == model_regst_);
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void NormalizationMdUpdtCompActor::Act() {
  const JobDesc* job_desc = JobDesc::Singleton();
  int64_t model_version_id = model_regst_->model_version_id();
  if (model_version_id == job_desc->TotalBatchNum()) {
    AsyncSendRegstMsgToConsumer([this](int64_t actor_id) {
      return actor_id == related_save_model_actor_id_;
    });
  } else {
    if (model_version_id % job_desc->NumOfBatchesInSnapshot() == 0) {
      AsyncSendRegstMsgToConsumer([this](int64_t actor_id) {
        return actor_id == related_save_model_actor_id_;
      });
    }
  }
}

REGISTER_ACTOR(TaskType::kNormalizationMdUpdt, NormalizationMdUpdtCompActor);

}  // namespace oneflow
