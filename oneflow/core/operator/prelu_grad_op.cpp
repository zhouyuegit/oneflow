#include "oneflow/core/operator/prelu_grad_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluGradOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_grad_conf());
  StrFieldTolower("data_format");
  EnrollInputBn("in");
  EnrollInputBn("out_diff");
  EnrollInputBn("alpha");
  EnrollOutputBn("alpha_diff");
  EnrollOutputBn("in_diff")->set_mutable_inplace_ibn("out_diff");
  if (device_type() == DeviceType::kGPU) { EnrollFwBufBn("fw_buf"); }
}

const PbMessage& PReluGradOp::GetCustomizedConf() const { return op_conf().prelu_grad_conf(); }

void PReluGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const PReluGradOpConf& conf = op_conf().prelu_grad_conf();
  const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp("out_diff");
  *GetBlobDesc4BnInOp("in_diff") = *out_diff_blob_desc;
}

void PReluGradOp::InferFwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  if (device_type() == DeviceType::kGPU) {
    BlobDesc* fw_buf_desc = GetBlobDesc4BnInOp("fw_buf");
    if (op_conf().prelu_grad_conf().channel_shared()) {
      *fw_buf_desc = *GetBlobDesc4BnInOp("out_diff");
    } else {
      const PReluGradOpConf& conf = op_conf().prelu_grad_conf();
      const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp("out_diff");
      fw_buf_desc->set_data_type(out_diff_blob_desc->data_type());
      std::vector<int64_t> fw_buf_shape_vec = out_diff_blob_desc->shape().dim_vec();
      if (conf.data_format() == "channels_first") {
        fw_buf_shape_vec[0] = out_diff_blob_desc->shape().At(1);
        fw_buf_shape_vec[1] = out_diff_blob_desc->shape().At(0);
        fw_buf_desc->mut_shape() = Shape(fw_buf_shape_vec);
      } else if (conf.data_format() == "channels_last") {
        fw_buf_shape_vec[0] = out_diff_blob_desc->shape().At(out_diff_blob_desc->shape().NumAxes() - 1);
        fw_buf_shape_vec[out_diff_blob_desc->shape().NumAxes() - 1] = out_diff_blob_desc->shape().At(0);
        fw_buf_desc->mut_shape() = Shape(fw_buf_shape_vec);
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

// TODO(shiyuan) ?
void PReluGradOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

void PReluGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PReluGradOpConf& conf = op_conf().prelu_grad_conf();
  PbRf<int32_t>* perm = kernel_conf->mutable_prelu_grad_conf()->mutable_perm();
  const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp("out_diff");
  int64_t num_axes = out_diff_blob_desc->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes) { perm->Add(i); }
  if (!conf.channel_shared()) {
    if (conf.data_format() == "channels_first") {
      (*perm)[0] = 1;
      (*perm)[1] = 0;
    } else if (conf.data_format() == "channels_last") {
      (*perm)[num_axes - 1] = 0;
      (*perm)[0] = num_axes - 1;
    } else {
      UNIMPLEMENTED();
    }
  }
}

void PReluGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  CHECK(*HasBatchDim4BnInOp("in"));
  CHECK(*HasBatchDim4BnInOp("out_diff"));
  *HasBatchDim4BnInOp("alpha_diff") = false;
  *HasBatchDim4BnInOp("in_diff") = *HasBatchDim4BnInOp("out_diff");
}

void PReluGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("in", 0)
      .Split("out_diff", 0)
      .Broadcast("alpha")
      .Broadcast("alpha_diff")
      .Split("in_diff", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kPreluGradConf, PReluGradOp);

}  // namespace oneflow
