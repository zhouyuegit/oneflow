#include "oneflow/core/job_completer/indexed_slices_optimizer_rewrite_pass.h"

namespace oneflow {

void IndexedSlicesOptimizerRewritePass::Apply(const OpGraph& op_graph,
                                              JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* src_node) {
    const OperatorConf& src_op_conf = src_node->op().op_conf();
    if (!src_op_conf.has_gather_ms0_grad_conf()) { return; }
    const GatherMs0GradOpConf& gather_ms0_grad_conf = src_op_conf.gather_ms0_grad_conf();
    if (src_node->out_edges().size() != 1) { return; }
    const OpNode* dst_node = src_node->SoleOutEdge()->dst_node();
    const OperatorConf& dst_op_conf = dst_node->op().op_conf();
    std::string indices_lbn = gather_ms0_grad_conf.indices();
    std::string values_lbn = gather_ms0_grad_conf.out_diff();
    if (dst_op_conf.has_lazy_adam_model_update_conf()) {
      const LazyAdamModelUpdateOpConf& old_optimizer_conf =
          dst_op_conf.lazy_adam_model_update_conf();
      const SbpParallel& model_sbp = dst_node->SbpParallel4BnInOp("model");
      if (!(model_sbp.has_broadcast_parallel()
            || (model_sbp.has_split_parallel() && model_sbp.split_parallel().axis() == 0))) {
        return;
      }
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      {
        const std::string& total_instance_num_lbn = old_optimizer_conf.total_instance_num_diff();
        const LogicalBlobId total_instance_num_lbi = GenLogicalBlobId(total_instance_num_lbn);
        const OpNode* total_instance_num_node =
            op_graph.OpNode4OpName(total_instance_num_lbi.op_name());
        if (total_instance_num_node->op().op_conf().has_constant_conf()) {
          const ConstantOpConf& constant_conf =
              total_instance_num_node->op().op_conf().constant_conf();
          float total_instance_num_scalar = 0;
          if (constant_conf.initializer().has_constant_int_conf()) {
            total_instance_num_scalar = constant_conf.initializer().constant_int_conf().value();
          } else if (constant_conf.initializer().has_constant_conf()) {
            total_instance_num_scalar = constant_conf.initializer().constant_conf().value();
          } else {
            UNIMPLEMENTED();
          }
          if (total_instance_num_scalar != 1.0f) {
            OperatorConf scalar_mul_op_conf{};
            scalar_mul_op_conf.set_name("System-Optimizer-IndexedSlices-" + model_lbi.op_name()
                                        + "-ScalarMul");
            ScalarMulOpConf* scalar_mul_conf = scalar_mul_op_conf.mutable_scalar_mul_conf();
            scalar_mul_conf->set_in(values_lbn);
            scalar_mul_conf->set_out("out");
            scalar_mul_conf->set_float_operand(1.0f / total_instance_num_scalar);
            values_lbn = GenLogicalBlobName(scalar_mul_op_conf.name(), scalar_mul_conf->out());
            job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {scalar_mul_op_conf});
          }
        } else {
          OperatorConf broadcast_div_op_conf{};
          broadcast_div_op_conf.set_name("System-Optimizer-IndexedSlices-" + model_lbi.op_name()
                                         + "-BroadcastDiv");
          BroadcastDivOpConf* broadcast_div_conf =
              broadcast_div_op_conf.mutable_broadcast_div_conf();
          broadcast_div_conf->set_a(values_lbn);
          broadcast_div_conf->set_b(old_optimizer_conf.total_instance_num_diff());
          broadcast_div_conf->set_out("out");
          values_lbn = GenLogicalBlobName(broadcast_div_op_conf.name(), broadcast_div_conf->out());
          job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {broadcast_div_op_conf});
        }
      }
      OperatorConf new_optimizer_op_conf{};
      {
        new_optimizer_op_conf.set_name("System-Optimizer-IndexedSlices-" + model_lbi.op_name());
        IndexedSlicesLazyAdamOptimizerOpConf* new_optimizer_conf =
            new_optimizer_op_conf.mutable_indexed_slices_lazy_adam_optimizer_conf();
        new_optimizer_conf->set_m(old_optimizer_conf.m());
        new_optimizer_conf->set_v(old_optimizer_conf.v());
        new_optimizer_conf->set_model_diff_indices(indices_lbn);
        new_optimizer_conf->set_model_diff_values(values_lbn);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_train_step(old_optimizer_conf.train_step());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
        new_optimizer_conf->set_l1(old_optimizer_conf.l1());
        new_optimizer_conf->set_l2(old_optimizer_conf.l2());
        new_optimizer_conf->set_beta1(old_optimizer_conf.user_conf().lazy_adam_conf().beta1());
        new_optimizer_conf->set_beta2(old_optimizer_conf.user_conf().lazy_adam_conf().beta2());
        new_optimizer_conf->set_epsilon(old_optimizer_conf.user_conf().lazy_adam_conf().epsilon());
      }
      job_builder->DelOps({src_op_conf, dst_op_conf});
      job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {new_optimizer_op_conf});
    } else {
      return;
    }
  });
}

}  // namespace oneflow
