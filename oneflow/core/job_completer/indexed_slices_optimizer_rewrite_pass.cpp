#include "oneflow/core/job_completer/indexed_slices_optimizer_rewrite_pass.h"

namespace oneflow {

void IndexedSlicesOptimizerRewritePass::Apply(const OpGraph& op_graph,
                                              JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* src_node) {
    const OperatorConf& src_op_conf = src_node->op().op_conf();
    if (src_node->out_edges().size() != 1) { return; }
    std::string indices_lbn;
    std::string values_lbn;
    std::string total_instance_num_diff_lbn;
    std::string model_op_name;
    std::function<void(OperatorConf * new_optimizer_op_conf, const std::string& indices,
                       const std::string& values)>
        BuildOptimizer;
    if (src_op_conf.has_gather_ms0_grad_conf()) {
      const GatherMs0GradOpConf& gather_ms0_grad_conf = src_op_conf.gather_ms0_grad_conf();
      indices_lbn = gather_ms0_grad_conf.indices();
      values_lbn = gather_ms0_grad_conf.out_diff();
    } else if (src_op_conf.has_unsorted_segment_sum_conf()) {
      const UnsortedSegmentSumOpConf& unsorted_segment_sum_conf =
          src_op_conf.unsorted_segment_sum_conf();
      if (unsorted_segment_sum_conf.axis() == 0) {
        indices_lbn = unsorted_segment_sum_conf.segment_ids();
        values_lbn = unsorted_segment_sum_conf.data();
      }
    } else {
      return;
    }
    const OpNode* dst_node = src_node->SoleOutEdge()->dst_node();
    const OperatorConf& dst_op_conf = dst_node->op().op_conf();
    if (dst_op_conf.has_naive_model_update_conf()) {
      const NaiveModelUpdateOpConf& old_optimizer_conf = dst_op_conf.naive_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      total_instance_num_diff_lbn = old_optimizer_conf.total_instance_num_diff();
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        OperatorConf neg_learning_rate_op_conf;
        neg_learning_rate_op_conf.set_name("System-Optimizer-IndexedSlices-NegLearningRate-"
                                           + model_op_name);
        ScalarMulOpConf* scalar_mul_op_conf = neg_learning_rate_op_conf.mutable_scalar_mul_conf();
        scalar_mul_op_conf->set_in(values);
        scalar_mul_op_conf->set_out("out");
        scalar_mul_op_conf->set_float_operand(-1.0f);

        OperatorConf apply_learning_rate_op_conf;
        apply_learning_rate_op_conf.set_name("System-Optimizer-IndexedSlices-ApplyLearningRate-"
                                             + model_op_name);
        BroadcastMulOpConf* broadcast_mul_op_conf =
            apply_learning_rate_op_conf.mutable_broadcast_mul_conf();
        broadcast_mul_op_conf->set_a(values);
        broadcast_mul_op_conf->set_b(
            GenLogicalBlobName(neg_learning_rate_op_conf.name(), scalar_mul_op_conf->out()));
        broadcast_mul_op_conf->set_out("out");

        ScatterAddOpConf* new_optimizer_conf = new_optimizer_op_conf->mutable_scatter_add_conf();
        new_optimizer_conf->set_indices(indices);
        new_optimizer_conf->set_updates(
            GenLogicalBlobName(apply_learning_rate_op_conf.name(), broadcast_mul_op_conf->out()));
        new_optimizer_conf->set_ref(old_optimizer_conf.model());

        job_builder->AddOps(dst_node->parallel_desc().parallel_conf(),
                            {neg_learning_rate_op_conf, apply_learning_rate_op_conf});
      };
    } else if (dst_op_conf.has_momentum_model_update_conf()) {
      const MomentumModelUpdateOpConf& old_optimizer_conf =
          dst_op_conf.momentum_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      total_instance_num_diff_lbn = old_optimizer_conf.total_instance_num_diff();
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        IndexedSlicesMomentumOptimizerOpConf* new_optimizer_conf =
            new_optimizer_op_conf->mutable_indexed_slices_momentum_optimizer_conf();
        new_optimizer_conf->set_momentum(old_optimizer_conf.momentum());
        new_optimizer_conf->set_model_diff_indices(indices);
        new_optimizer_conf->set_model_diff_values(values);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_train_step(old_optimizer_conf.train_step());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
        new_optimizer_conf->set_l1(old_optimizer_conf.l1());
        new_optimizer_conf->set_l2(old_optimizer_conf.l2());
        new_optimizer_conf->set_beta(old_optimizer_conf.user_conf().momentum_conf().beta());
      };
    } else if (dst_op_conf.has_lazy_adam_model_update_conf()) {
      const LazyAdamModelUpdateOpConf& old_optimizer_conf =
          dst_op_conf.lazy_adam_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      total_instance_num_diff_lbn = old_optimizer_conf.total_instance_num_diff();
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        IndexedSlicesLazyAdamOptimizerOpConf* new_optimizer_conf =
            new_optimizer_op_conf->mutable_indexed_slices_lazy_adam_optimizer_conf();
        new_optimizer_conf->set_m(old_optimizer_conf.m());
        new_optimizer_conf->set_v(old_optimizer_conf.v());
        new_optimizer_conf->set_model_diff_indices(indices);
        new_optimizer_conf->set_model_diff_values(values);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_train_step(old_optimizer_conf.train_step());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
        new_optimizer_conf->set_l1(old_optimizer_conf.l1());
        new_optimizer_conf->set_l2(old_optimizer_conf.l2());
        new_optimizer_conf->set_beta1(old_optimizer_conf.user_conf().lazy_adam_conf().beta1());
        new_optimizer_conf->set_beta2(old_optimizer_conf.user_conf().lazy_adam_conf().beta2());
        new_optimizer_conf->set_epsilon(old_optimizer_conf.user_conf().lazy_adam_conf().epsilon());
      };
    } else {
      return;
    }
    if (!BuildOptimizer) { return; }
    CHECK(!total_instance_num_diff_lbn.empty());
    CHECK(!indices_lbn.empty());
    CHECK(!values_lbn.empty());
    CHECK(!model_op_name.empty());
    const LogicalBlobId total_instance_num_lbi = GenLogicalBlobId(total_instance_num_diff_lbn);
    const OpNode* total_instance_num_node =
        op_graph.OpNode4OpName(total_instance_num_lbi.op_name());
    if (total_instance_num_node->op().op_conf().has_constant_conf()) {
      const ConstantOpConf& constant_conf = total_instance_num_node->op().op_conf().constant_conf();
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
        scalar_mul_op_conf.set_name("System-Optimizer-IndexedSlices-ScalarMul-" + model_op_name);
        ScalarMulOpConf* scalar_mul_conf = scalar_mul_op_conf.mutable_scalar_mul_conf();
        scalar_mul_conf->set_in(values_lbn);
        scalar_mul_conf->set_out("out");
        scalar_mul_conf->set_float_operand(1.0f / total_instance_num_scalar);
        values_lbn = GenLogicalBlobName(scalar_mul_op_conf.name(), scalar_mul_conf->out());
        job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {scalar_mul_op_conf});
      }
    } else {
      OperatorConf broadcast_div_op_conf{};
      broadcast_div_op_conf.set_name("System-Optimizer-IndexedSlices-BroadcastDiv-"
                                     + model_op_name);
      BroadcastDivOpConf* broadcast_div_conf = broadcast_div_op_conf.mutable_broadcast_div_conf();
      broadcast_div_conf->set_a(values_lbn);
      broadcast_div_conf->set_b(total_instance_num_diff_lbn);
      broadcast_div_conf->set_out("out");
      values_lbn = GenLogicalBlobName(broadcast_div_op_conf.name(), broadcast_div_conf->out());
      job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {broadcast_div_op_conf});
    }
    OperatorConf new_optimizer_op_conf{};
    new_optimizer_op_conf.set_name("System-Optimizer-IndexedSlices-" + model_op_name);
    BuildOptimizer(&new_optimizer_op_conf, indices_lbn, values_lbn);
    job_builder->DelOps({src_op_conf, dst_op_conf});
    job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {new_optimizer_op_conf});
  });
}

}  // namespace oneflow
