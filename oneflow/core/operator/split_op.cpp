#include "oneflow/core/operator/split_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void SplitOp::InitFromOpConf() {
  CHECK(op_conf().has_split_conf());

  EnrollInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& SplitOp::GetCustomizedConf() const { return op_conf().split_conf(); }

void SplitOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const SplitOpConf& conf = op_conf().split_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> out_dim_vec = in_blob_desc->shape().dim_vec();
  int32_t split_axis = conf.axis();
  if (split_axis < 0) { split_axis += out_dim_vec.size(); }

  BalancedSplitter splitter(out_dim_vec[split_axis], output_bns().size());
  for (size_t i = 0; i < output_bns().size(); ++i) {
    BlobDesc* out_i_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    for (int64_t j = 0; j < out_dim_vec.size(); ++j) {
      if (j == split_axis) {
        out_dim_vec[j] = splitter.At(i).size();
      } else {
        CHECK_EQ(out_dim_vec[j], in_blob_desc->shape().At(j));
      }
    }
    *out_i_blob_desc = *in_blob_desc;
    out_i_blob_desc->mut_shape() = Shape(out_dim_vec);
  }
  int64_t total_length = 0;
  for (size_t i = 0; i < output_bns().size(); ++i) {
    BlobDesc* out_i_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    total_length += out_i_blob_desc->shape().At(split_axis);
  }
  CHECK_EQ(total_length, in_blob_desc->shape().At(split_axis));
}

REGISTER_OP(OperatorConf::kSplitConf, SplitOp);

}  // namespace oneflow
