#include "oneflow/core/actor/normal_forward_compute_actor.h"

namespace oneflow {

void NormalForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  cur_piece_id_ = -1;
  const_buf_regst_desc_id_ = Name2SoleRegstDescId("const_buf");
  const_buf_regst_ = nullptr;
  if (const_buf_regst_desc_id_ != -1) {
    const_buf_regst_ = GetSoleProducedRegst4RegstDescId(const_buf_regst_desc_id_);
  }
  send_const_buf_regst_ = false;
  if (const_buf_regst_desc_id_ != -1) { SendConstBufInitMsgToBwActor(); }
  OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
}

bool NormalForwardCompActor::IsCustomizedWriteReady() const {
  if (const_buf_regst_desc_id_ != -1) { CHECK(send_const_buf_regst_); }
  return true;
}

void NormalForwardCompActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(const_buf_regst_, regst);
  const_buf_regst_ = nullptr;
  send_const_buf_regst_ = false;
}

bool NormalForwardCompActor::CheckOutputActId(int64_t regst_desc_id) const { return true; }

void NormalForwardCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {}

void NormalForwardCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  cur_piece_id_ = GetPieceId4NaiveOrInplaceCurReadableDataRegst();
  std::tuple<int64_t, std::function<const Blob*(const LogicalBlobId&)>> other_val(
      cur_piece_id_, [](const LogicalBlobId& lbi) -> const Blob* { return nullptr; });
  kernel_ctx.other = &other_val;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == const_buf_regst_desc_id_) {
      return const_buf_regst_;
    } else {
      return nullptr;
    }
  });
}

void NormalForwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
}

void NormalForwardCompActor::VirtualAsyncSendInplaceProducedRegstMsgToConsumer() {
  HandleProducedInplaceDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
}

void NormalForwardCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() { cur_piece_id_ = -1; }

void NormalForwardCompActor::SendConstBufInitMsgToBwActor() {
  CHECK_EQ(0, ReadingCnt4ProducedRegst(const_buf_regst_));
  const_buf_regst_->set_act_id(act_id());
  for (int64_t consumer : const_buf_regst_->consumers_actor_id()) {
    EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, const_buf_regst_));
  }
  IncreaseReadingCnt4ProducedRegst(const_buf_regst_, const_buf_regst_->consumers_actor_id().size());
  IncreaseTotalReadingCnt(const_buf_regst_->consumers_actor_id().size());
  send_const_buf_regst_ = true;
}

REGISTER_ACTOR(TaskType::kNormalForward, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kAccuracy, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kOptimizer, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kPrint, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kForeignInput, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kForeignOutput, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kDistributeConcat, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kDistributeSplit, NormalForwardCompActor);

}  // namespace oneflow
