#include "oneflow/core/actor/evaluator_data_loader_actor.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace {

void RandInitBlob(Blob* blob) {
  // TODO, only support float and GPU here
  RandomGenerator rng(GetCurTime());
  rng.Uniform<kGPU, float>(blob->shape().elem_cnt(), blob->mut_dptr<float>());
}

}  // namespace

void EvalDataLdActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_eof_ = false;
  sudo_piece_num_ = 2;
  for (const auto& pair : task_proto.produced_regst_desc()) {
    Regst* regst = GetCurWriteableRegst(pair.second.regst_desc_id());
    for (const auto& pair : regst->lbi2blob()) {
      RandInitBlob(static_cast<Blob*>(pair.second.get()));
    }
  }
  OF_SET_MSG_HANDLER(&EvalDataLdActor::HandlerNormal);
}

void EvalDataLdActor::Act() {
  sudo_piece_num_ -= 1;
  if (sudo_piece_num_ == 0) { is_eof_ = true; }
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(0);
    return true;
  });
}

bool EvalDataLdActor::IsCustomizedReadReady() { return !is_eof_; }

REGISTER_ACTOR(TaskType::kEvalDataLd, EvalDataLdActor);

}  // namespace oneflow
