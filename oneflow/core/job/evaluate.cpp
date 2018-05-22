#include "oneflow/core/job/runtime.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd) {
  for (const TaskProto* task : tasks) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(task->task_id(), cmd);
    Global<ActorMsgBus>::Get()->SendMsg(msg);
  }
}

void HandoutTasks(const std::vector<const TaskProto*>& tasks) {
  for (const TaskProto* task : tasks) { Global<ThreadMgr>::Get()->GetThrd(0)->AddTask(*task); }
  SendCmdMsg(tasks, ActorCmd::kConstructActor);
}

bool IsInVector(const int64_t target, const std::vector<int64_t>& vec) {
  return std::find(vec.begin(), vec.end(), target) != vec.end();
}

void RemoveIdsNotInEval(RegstDescProto& regst_desc, const std::vector<int64_t>& actor_ids) {
  std::vector<int64_t> id_tmp;
  id_tmp.clear();
  for (const int64_t id : regst_desc.consumer_task_id()) {
    if (!IsInVector(id, actor_ids)) { continue; }
    id_tmp.push_back(id);
  }
  regst_desc.clear_consumer_task_id();
  for (const int64_t id : id_tmp) { regst_desc.add_consumer_task_id(id); }
}

void CreateEvalProducerTaskProto(const std::vector<int64_t>& actor_ids,
                                 const std::vector<int64_t>& consumed_regst_desc_id_vec,
                                 const Plan& raw_plan, Plan& eval_plan) {
  HashMap<int64_t, std::vector<int64_t>> producer_map;
  for (const TaskProto& task : raw_plan.task()) {
    if (IsInVector(task.task_id(), actor_ids)) { continue; }
    for (const auto& pair : task.produced_regst_desc()) {
      if (!IsInVector(pair.second.regst_desc_id(), consumed_regst_desc_id_vec)) { continue; }
      if (producer_map.find(task.task_id()) == producer_map.end()) {
        CHECK(producer_map.insert({task.task_id(), {pair.second.regst_desc_id()}}).second);
      } else {
        producer_map[task.task_id()].push_back(pair.second.regst_desc_id());
      }
    }
  }
  for (const TaskProto& task : raw_plan.task()) {
    if (producer_map.find(task.task_id()) == producer_map.end()) { continue; }
    auto begin_task_proto = eval_plan.mutable_task()->Add();
    if (task.task_type() == TaskType::kNormalMdUpdt) {
      begin_task_proto->set_task_type(TaskType::kEvalMdUpdt);
    } else {
      begin_task_proto->set_task_type(TaskType::kEvalDataLd);
    }
    begin_task_proto->set_machine_id(0);
    begin_task_proto->set_thrd_id(0);
    begin_task_proto->set_task_id(task.task_id());
    auto produced_regst_proto = begin_task_proto->mutable_produced_regst_desc();
    for (auto pair : task.produced_regst_desc()) {
      if (!IsInVector(pair.second.regst_desc_id(), producer_map[task.task_id()])) { continue; }
      // some consumer task id may be not the actors under evaluation, should be removed.
      RemoveIdsNotInEval(pair.second, actor_ids);
      CHECK(produced_regst_proto->insert(pair).second);
    }
  }
}

void CreateEvalConsumerTaskProto(const std::vector<int64_t> actor_ids,
                                 HashMap<int64_t, std::vector<int64_t>>& consumer_map,
                                 const Plan& raw_plan, Plan& eval_plan) {
  for (const TaskProto& task : raw_plan.task()) {
    if (consumer_map.find(task.task_id()) == consumer_map.end()) { continue; }
    auto end_task_proto = eval_plan.mutable_task()->Add();
    end_task_proto->set_task_type(TaskType::kEvalConsumer);
    end_task_proto->set_machine_id(0);
    end_task_proto->set_thrd_id(0);
    end_task_proto->set_task_id(task.task_id());
    auto consumed_regst_proto = end_task_proto->mutable_consumed_regst_desc_id();
    for (const auto& pair : task.consumed_regst_desc_id()) {
      // TBD, only compare elem 0
      if (!IsInVector(pair.second.regst_desc_id()[0], consumer_map[task.task_id()])) { continue; }
      CHECK(consumed_regst_proto->insert(pair).second);
    }
  }
}

}  // namespace

class Evaluator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Evaluator);
  ~Evaluator() = default;

  Evaluator(const JobDescProto& job_desc, const Plan& raw_plan,
            const std::vector<int64_t>& actor_ids);

 private:
  void NewAllGlobal(const JobDescProto& job_desc);
  void DeleteAllGlobal();
};

Evaluator::Evaluator(const JobDescProto& job_desc, const Plan& raw_plan,
                     const std::vector<int64_t>& actor_ids) {
  NewAllGlobal(job_desc);
  std::vector<const TaskProto*> other_tasks;
  std::vector<const TaskProto*> datald_tasks;
  std::vector<const TaskProto*> model_tasks;
  std::vector<int64_t> consumed_regst_desc_id_vec;
  HashMap<int64_t, std::vector<int64_t>> consumer_map;
  Plan eval_plan;
  int64_t this_machine_task_num = 0;

  // create evaluation plan
  for (const TaskProto& task : raw_plan.task()) {
    if (task.machine_id() != 0) { continue; }
    if (!IsInVector(task.task_id(), actor_ids)) { continue; }
    auto eval_task_proto = eval_plan.mutable_task()->Add();
    *eval_task_proto = task;
    for (const auto& pair : task.consumed_regst_desc_id()) {
      for (const int64_t id : pair.second.regst_desc_id()) {
        if (IsInVector(id, consumed_regst_desc_id_vec)) { continue; }
        consumed_regst_desc_id_vec.push_back(id);
      }
    }
    for (const auto& pair : task.produced_regst_desc()) {
      for (const auto& id : pair.second.consumer_task_id()) {
        if (consumer_map.find(id) == consumer_map.end()) {
          CHECK(consumer_map.insert({id, {pair.second.regst_desc_id()}}).second);
        } else {
          consumer_map[id].push_back(pair.second.regst_desc_id());
        }
      }
    }
  }
  CreateEvalProducerTaskProto(actor_ids, consumed_regst_desc_id_vec, raw_plan, eval_plan);
  CreateEvalConsumerTaskProto(actor_ids, consumer_map, raw_plan, eval_plan);
  PrintProtoToTextFile(eval_plan, JoinPath(LogDir(), "eval_plan"));

  // run the evaluation plan
  for (const TaskProto& task : eval_plan.task()) {
    if (task.task_type() == TaskType::kEvalDataLd) {
      datald_tasks.push_back(&task);
    } else if (task.task_type() == TaskType::kEvalMdUpdt) {
      model_tasks.push_back(&task);
    } else {
      other_tasks.push_back(&task);
    }
    this_machine_task_num += 1;
  }
  RuntimeCtx* runtime_ctx = Global<RuntimeCtx>::Get();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(datald_tasks);
  HandoutTasks(model_tasks);
  HandoutTasks(other_tasks);
  runtime_ctx->WaitUntilCntEqualZero("constructing_actor_cnt");
  LOG(INFO) << "All actor on this machine are constructed";

  DeleteAllGlobal();
}

void Evaluator::NewAllGlobal(const JobDescProto& job_desc) {
  int64_t piece_num = 0;
  Global<JobDesc>::New(job_desc);
  Global<RuntimeCtx>::New(piece_num, false);
  Global<ThreadMgr>::New(true);
  Global<ActorMsgBus>::New(true);
  Global<MemoryAllocator>::New();
  Global<RegstMgr>::New();
  Global<IDMgr>::New();
}

void Evaluator::DeleteAllGlobal() {
  Global<JobDesc>::Delete();
  Global<RuntimeCtx>::Delete();
  Global<ThreadMgr>::Delete();
  Global<ActorMsgBus>::Delete();
  Global<MemoryAllocator>::Delete();
  Global<RegstMgr>::Delete();
  Global<IDMgr>::Delete();
}

}  // namespace oneflow

DEFINE_string(plan_filepath, "", "");
DEFINE_string(actor_id, "", "");
DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(job_desc_filepath, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->RecursivelyCreateDirIfNotExist(LogDir());
  RedirectStdoutAndStderrToGlogDir();
  LOG(INFO) << "Evaluation Starting Up";

  Plan raw_plan;
  LOG(INFO) << "Parse Plan File";
  ParseProtoFromTextFile(FLAGS_plan_filepath, &raw_plan);
  JobDescProto job_desc;
  if (FLAGS_job_desc_filepath != "") {
    ParseProtoFromTextFile(FLAGS_job_desc_filepath, &job_desc);
  } else if (FLAGS_job_conf_filepath != "") {
    JobConf* jc = job_desc.mutable_job_conf();
    ParseProtoFromTextFile(FLAGS_job_conf_filepath, jc);
    ParseProtoFromTextFile(jc->dlnet_filepath(), job_desc.mutable_dlnet_conf());
    ParseProtoFromTextFile(jc->resource_filepath(), job_desc.mutable_resource());
    ParseProtoFromTextFile(jc->placement_filepath(), job_desc.mutable_placement());
    std::vector<int64_t> actor_ids;
    actor_ids.push_back(std::stoi(FLAGS_actor_id));
    Evaluator eval(job_desc, raw_plan, actor_ids);
  } else {
    LOG(FATAL) << "Please Set job_conf_filepath or job_desc_filepath";
  }

  LOG(INFO) << "Evaluation Shutting Down";
  CloseStdoutAndStderr();
  return 0;
}
