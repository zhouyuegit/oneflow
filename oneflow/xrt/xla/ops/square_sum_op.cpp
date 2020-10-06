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
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

xla::XlaOp Square(const xla::XlaOp &x) {
  return xla::Dot(x,x);
}

class SquareSumOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp x = ctx->Input("x");
    xla::XlaBuilder *builder = ctx->builder(); 
    DataType data_type = ctx->SoleInputType();
    xla::XlaOp sum;
    
    xla::XlaComputation add_func = CreateAddFunc(data_type);
    sum = xla::Reduce(Square(x), Zero(builder, data_type), add_func, {1});
    ctx->SetSoleOutput(sum);
  }
};

REGISTER_XLA_OP_KERNEL(SquareSum, SquareSumOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
