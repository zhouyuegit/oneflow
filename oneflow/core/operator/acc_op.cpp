/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/acc_op.h"

namespace oneflow {

void AccOp::InitFromOpConf() {
  CHECK(op_conf().has_acc_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

Maybe<void> AccOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("acc") = *GetBlobDesc4BnInOp("one");
  return Maybe<void>::Ok();
}

Maybe<void> AccOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int32_t max_acc_num = op_conf().acc_conf().max_acc_num();
  CHECK_GE_OR_RETURN(GetTimeShape4BnInOp("one")->elem_cnt(), max_acc_num);
  *time_shape = Shape({GetTimeShape4BnInOp("one")->elem_cnt() / max_acc_num});
  return Maybe<void>::Ok();
}

Maybe<void> AccOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("acc")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> AccOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  const SbpParallel sbp_parallel = JUST(SbpInferHint4Ibn("one"))->sbp_parallel();
  (*bn2sbp)["one"] = sbp_parallel;
  (*bn2sbp)["acc"] = sbp_parallel;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccConf, AccOp);

}  // namespace oneflow
