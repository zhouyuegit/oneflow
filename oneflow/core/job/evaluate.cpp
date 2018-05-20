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

void CreateBeginTasksMap(HashMap<int64_t, std::vector<int64_t>>& begin_map,
                         const std::vector<int64_t>& actor_ids,
                         const std::vector<int64_t>& consumed_regst_desc_id_vec, const Plan& plan) {
  for (const TaskProto& task : plan.task()) {
    if (IsInVector(task.task_id(), actor_ids)) { continue; }
    for (const auto& pair : task.produced_regst_desc()) {
      if (!IsInVector(pair.second.regst_desc_id(), consumed_regst_desc_id_vec)) { continue; }
      if (begin_map.find(task.task_id()) == begin_map.end()) {
        CHECK(begin_map.insert({task.task_id(), {pair.second.regst_desc_id()}}).second);
      } else {
        begin_map[task.task_id()].push_back(pair.second.regst_desc_id());
      }
    }
  }
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

void CreateBeginTasksProto(HashMap<int64_t, std::vector<int64_t>>& begin_map,
                           const std::vector<int64_t>& actor_ids, const Plan& plan,
                           Plan& eval_plan) {
  for (const TaskProto& task : plan.task()) {
    if (begin_map.find(task.task_id()) == begin_map.end()) { continue; }
    auto begin_task_proto = eval_plan.mutable_task()->Add();
    begin_task_proto->set_task_type(TaskType::kEvalBegin);
    begin_task_proto->set_machine_id(0);
    begin_task_proto->set_thrd_id(0);
    begin_task_proto->set_task_id(task.task_id());
    auto produced_regst_proto = begin_task_proto->mutable_produced_regst_desc();
    for (auto pair : task.produced_regst_desc()) {
      if (!IsInVector(pair.second.regst_desc_id(), begin_map[task.task_id()])) { continue; }
      // some consumer task id may be not the actors under evaluation, should be removed.
      RemoveIdsNotInEval(pair.second, actor_ids);
      CHECK(produced_regst_proto->insert(pair).second);
    }
  }
}

}  // namespace

class Evaluator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Evaluator);
  ~Evaluator() = default;

  Evaluator(const JobDescProto& job_desc, const Plan& plan, const std::vector<int64_t>& actor_ids);

 private:
  void NewAllGlobal(const JobDescProto& job_desc);
  void DeleteAllGlobal();
};

Evaluator::Evaluator(const JobDescProto& job_desc, const Plan& plan,
                     const std::vector<int64_t>& actor_ids) {
  NewAllGlobal(job_desc);
  std::vector<const TaskProto*> dut_tasks;
  std::vector<int64_t> consumed_regst_desc_id_vec;
  std::vector<int64_t> produced_regst_desc_id_vec;
  std::vector<const TaskProto*> begin_eval_tasks;
  std::vector<const TaskProto*> end_eval_tasks;
  HashMap<int64_t, std::vector<int64_t>> begin_task2produced_regsts_map;
  int64_t this_machine_task_num = 0;

  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != 0) { continue; }
    if (IsInVector(task.task_id(), actor_ids)) {
      dut_tasks.push_back(&task);
      this_machine_task_num += 1;
      for (const auto& pair : task.consumed_regst_desc_id()) {
        for (int64_t id : pair.second.regst_desc_id()) {
          if (!IsInVector(id, consumed_regst_desc_id_vec)) {
            consumed_regst_desc_id_vec.push_back(id);
          }
        }
      }
      for (const auto& pair : task.produced_regst_desc()) {
        produced_regst_desc_id_vec.push_back(pair.second.regst_desc_id());
      }
    }
  }
  CreateBeginTasksMap(begin_task2produced_regsts_map, actor_ids, consumed_regst_desc_id_vec, plan);
  Plan eval_plan;
  CreateBeginTasksProto(begin_task2produced_regsts_map, actor_ids, plan, eval_plan);
  PrintProtoToTextFile(eval_plan, JoinPath(LogDir(), "eval_plan"));
  RuntimeCtx* runtime_ctx = Global<RuntimeCtx>::Get();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(dut_tasks);
  runtime_ctx->WaitUntilCntEqualZero("constructing_actor_cnt");
  LOG(INFO) << "All actor on this machine are constructed";

  for (auto& eval_task : begin_eval_tasks) { delete eval_task; }
  for (auto& eval_task : end_eval_tasks) { delete eval_task; }
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

  Plan plan;
  LOG(INFO) << "Parse Plan File";
  ParseProtoFromTextFile(FLAGS_plan_filepath, &plan);
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
    Evaluator eval(job_desc, plan, actor_ids);
  } else {
    LOG(FATAL) << "Please Set job_conf_filepath or job_desc_filepath";
  }

  LOG(INFO) << "Evaluation Shutting Down";
  CloseStdoutAndStderr();
  return 0;
}
