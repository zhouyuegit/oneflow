#include "oneflow/core/operator/mean_squared_loss_op.h"

namespace oneflow {

void MeanSquaredLossOp::VirtualInitFromOpConf() { EnrollDataTmpBn("diff"); }

const PbMessage& MeanSquaredLossOp::GetCustomizedConf() const {
  return op_conf().mean_squared_loss_conf();
}

LossKernelConf* MeanSquaredLossOp::GetMutLossKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_mean_squared_loss_conf()->mutable_loss_conf();
}

void MeanSquaredLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* diff_blob_desc = GetBlobDesc4BnInOp("diff");
  diff_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  diff_blob_desc->set_data_type(pred_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kMeanSquaredLossConf, MeanSquaredLossOp);

}  // namespace oneflow
