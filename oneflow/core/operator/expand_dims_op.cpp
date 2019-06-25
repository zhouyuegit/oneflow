#include "oneflow/core/operator/expand_dims_op.h"

namespace oneflow {

void ExpandDimsOp::InitFromOpConf() {
  CHECK(op_conf().has_expand_dims_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& ExpandDimsOp::GetCustomizedConf() const { return op_conf().expand_dims_conf(); }

void ExpandDimsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  std::vector<int64_t> dim_vec = in_blob_desc->shape().dim_vec();
  int32_t dim = op_conf().expand_dims_conf().dim();
  CHECK_GE(dim, -static_cast<int32_t>(dim_vec.size()) - 1);
  CHECK_LE(dim, dim_vec.size());
  std::vector<int64_t>::iterator it;
  if (dim >= 0) {
    it = dim_vec.begin() + dim;
  } else {
    it = dim_vec.end() + 1 + dim;
  }
  dim_vec.insert(it, 1);
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
}

void ExpandDimsOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
    .Split("in", 0)
    .Split("out", 0)
    .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kExpandDimsConf, ExpandDimsOp);

}  // namespace oneflow
