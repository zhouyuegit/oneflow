#include "oneflow/core/actor/evaluator_consumer_actor.h"

namespace oneflow {

void EvalConsumerActor::VirtualCompActorInit(const TaskProto& task_proto) {
  const_buf_regst_ = nullptr;
  const_buf_regst_desc_id_ = -1;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    if (pair.first != "const_buf") { continue; }
    // this actor add any consumed regst to naive_readable_regst_ in the
    // TakeOverNaiveConsumed in base Init(), so here remove the const buf reg.
    for (int64_t const_buf_id : pair.second.regst_desc_id()) {
      RemoveConstBufFromNaiveConsumedRegst(const_buf_id);
      const_buf_regst_desc_id_ = const_buf_id;
    }
  }
  OF_SET_MSG_HANDLER(&EvalConsumerActor::HandlerNormal);
}

void EvalConsumerActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(msg.regst()->regst_desc_id(), const_buf_regst_desc_id_);
  CHECK(const_buf_regst_ == nullptr);
  const_buf_regst_ = msg.regst();
}

void EvalConsumerActor::AsyncReturnAllCustomizedReadableRegst() {
  if (const_buf_regst_) {
    AsyncSendRegstMsgToProducer(const_buf_regst_);
    const_buf_regst_ = nullptr;
  } else {
    // not has const buf, do nothing
  }
}

REGISTER_ACTOR(TaskType::kEvalConsumer, EvalConsumerActor);

}  // namespace oneflow
