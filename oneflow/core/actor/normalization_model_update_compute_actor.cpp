#include "oneflow/core/actor/normalization_model_update_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void NormalizationMdUpdtCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  related_save_model_actor_id_ = task_proto.related_save_model_task_id();
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  // norm_model_regst_desc_id_ = RegstDescId4Name("norm_model");
  is_norm_acc_eord_ = false;
  acc_cnt_ = 0;
  max_acc_cnt_ = JobDesc::Singleton()->NumOfPiecesInBatch();
  readable_regst_mgr_.Init(task_proto);
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
    InitRegstBySendToFw(RegstDescId4Name("norm_model"));
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
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(0);
    return true;
  });
  OF_SET_MSG_HANDLER(&NormalizationMdUpdtCompActor::HandlerNormal);
  return 0;
}

int NormalizationMdUpdtCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kEordMsg) {
    is_norm_acc_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = actor_msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      readable_regst_mgr_.Push(regst);
    }
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void NormalizationMdUpdtCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return readable_regst_mgr_.GetCurReadable(regst_desc_id);
    } else {
      return regst;
    }
  });
  acc_cnt_ += 1;
  if (acc_cnt_ == max_acc_cnt_) {
    AsyncSendRegstMsgToConsumer([this](int64_t actor_id) {
      return actor_id == related_save_model_actor_id_;
    });
    acc_cnt_ = 0;
  }
  readable_regst_mgr_.ReturnToProducerAndPopCurReadable(this);
}

bool NormalizationMdUpdtCompActor::IsReadReady() {
  return readable_regst_mgr_.IsReadReady();
}

bool NormalizationMdUpdtCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_norm_acc_eord_ && readable_regst_mgr_.IsEmpty();
}

void NormalizationMdUpdtCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> func) {
  readable_regst_mgr_.ForEachCurReadableRegst(func);
}

REGISTER_ACTOR(TaskType::kNormalizationMdUpdt, NormalizationMdUpdtCompActor);

}  // namespace oneflow
