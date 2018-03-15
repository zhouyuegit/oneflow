#ifndef ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class NormalizationMdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationMdUpdtCompActor);
  NormalizationMdUpdtCompActor() = default;
  ~NormalizationMdUpdtCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  void InitRegstBySendToFw(int64_t regst_desc_id);

  int HandlerInitModel(const ActorMsg&);
  int HandlerSendInitialModel(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return false; }
  bool IsReadAlwaysUnReadyFromNow() override { return true; }
  void AsyncReturnAllReadableRegst() override {}

  int64_t model_regst_desc_id_;
  int64_t related_save_model_actor_id_;
  int64_t related_init_model_actor_id_;
  Regst* model_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_
