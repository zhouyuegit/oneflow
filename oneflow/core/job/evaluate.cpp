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

}  // namespace

class Evaluator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Evaluator);
  ~Evaluator() = default;

  Evaluator(const JobDescProto& job_desc, const Plan& plan, const int64_t actor_id);

 private:
  void NewAllGlobal(const JobDescProto& job_desc);
  void DeleteAllGlobal();
};

Evaluator::Evaluator(const JobDescProto& job_desc, const Plan& plan, const int64_t actor_id) {
  NewAllGlobal(job_desc);
  std::vector<const TaskProto*> eval_tasks;
  int64_t this_machine_task_num = 0;

  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != 0) { continue; }
    if (task.task_id() == actor_id) {
      eval_tasks.push_back(&task);
      this_machine_task_num += 1;
    }
  }
  RuntimeCtx* runtime_ctx = Global<RuntimeCtx>::Get();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(eval_tasks);
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
    Evaluator eval(job_desc, plan, std::stoi(FLAGS_actor_id));
  } else {
    LOG(FATAL) << "Please Set job_conf_filepath or job_desc_filepath";
  }

  LOG(INFO) << "Evaluation Shutting Down";
  CloseStdoutAndStderr();
  return 0;
}
