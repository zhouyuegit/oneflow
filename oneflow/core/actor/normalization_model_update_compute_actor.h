#ifndef ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/actor/naive_readable_register_manager.h"

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
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  // int64_t norm_model_regst_desc_id_;
  int64_t related_save_model_actor_id_;
  int64_t related_init_model_actor_id_;
  NaiveReadableRegstMgr readable_regst_mgr_;
  bool is_norm_acc_eord_;
  int32_t max_acc_cnt_;
  int32_t acc_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMALIZATION_MODEL_UPDATE_COMPUTE_ACTOR_H_
