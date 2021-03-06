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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> CheckTensorDescLike(const user_op::TensorDesc* tensor_desc,
                                const user_op::TensorDesc* like) {
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), like->data_type());
  CHECK_EQ_OR_RETURN(tensor_desc->shape(), like->shape());
  return Maybe<void>::Ok();
}

Maybe<void> CheckScalarTensorDesc(const user_op::TensorDesc* tensor_desc,
                                  const DataType data_type) {
  CHECK_EQ_OR_RETURN(tensor_desc->shape(), Shape({1}));
  CHECK_EQ_OR_RETURN(tensor_desc->data_type(), data_type);
  return Maybe<void>::Ok();
}

Maybe<void> CheckLearningRateTenserDesc(const user_op::TensorDesc* learning_rate) {
  JUST(CheckScalarTensorDesc(learning_rate, DataType::kFloat));
  return Maybe<void>::Ok();
}

Maybe<void> CheckIndexedSlicesModelDiffDesc(const user_op::TensorDesc* model,
                                            const user_op::TensorDesc* model_diff_indices,
                                            const user_op::TensorDesc* model_diff_values) {
  CHECK_OR_RETURN(IsIndexDataType(model_diff_indices->data_type()));
  const int64_t num_indices_axes = model_diff_indices->shape().NumAxes();
  const int64_t num_values_axes = model_diff_values->shape().NumAxes();
  CHECK_GE_OR_RETURN(num_values_axes, num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ_OR_RETURN(model_diff_values->shape().At(i), model_diff_indices->shape().At(i));
  }
  CHECK_EQ_OR_RETURN(model->data_type(), model_diff_values->data_type());
  const int64_t num_model_axes = model->shape().NumAxes();
  CHECK_EQ_OR_RETURN(num_model_axes, num_values_axes - num_indices_axes + 1);
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    CHECK_EQ_OR_RETURN(model->shape().At(i),
                       model_diff_values->shape().At(num_indices_axes + i - 1));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const Shape& shape = model->shape();
  const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
    const auto* scale_by_tensor = ctx->TensorDesc4ArgNameAndIndex("scale_by_tensor", 0);
    JUST(CheckScalarTensorDesc(scale_by_tensor, model->data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferIndexedSlicesSGDUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const user_op::TensorDesc* model_diff_indices =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
  const user_op::TensorDesc* model_diff_values =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(model, model_diff_indices, model_diff_values));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  return Maybe<void>::Ok();
}

Maybe<void> InferMomentumUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff->shape(), model->shape());
  const user_op::TensorDesc* momentum = ctx->TensorDesc4ArgNameAndIndex("momentum", 0);
  JUST(CheckTensorDescLike(momentum, model));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
    const auto* scale_by_tensor = ctx->TensorDesc4ArgNameAndIndex("scale_by_tensor", 0);
    JUST(CheckScalarTensorDesc(scale_by_tensor, model->data_type()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferIndexedSlicesMomentumUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const user_op::TensorDesc* model_diff_indices =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
  const user_op::TensorDesc* model_diff_values =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(model, model_diff_indices, model_diff_values));
  const user_op::TensorDesc* momentum = ctx->TensorDesc4ArgNameAndIndex("momentum", 0);
  JUST(CheckTensorDescLike(momentum, model));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  return Maybe<void>::Ok();
}

Maybe<void> InferAdamUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const DataType data_type = model->data_type();
  const Shape& shape = model->shape();
  const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
  const user_op::TensorDesc* m = ctx->TensorDesc4ArgNameAndIndex("m", 0);
  JUST(CheckTensorDescLike(m, model));
  const user_op::TensorDesc* v = ctx->TensorDesc4ArgNameAndIndex("v", 0);
  JUST(CheckTensorDescLike(v, model));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
    const auto* scale_by_tensor = ctx->TensorDesc4ArgNameAndIndex("scale_by_tensor", 0);
    JUST(CheckScalarTensorDesc(scale_by_tensor, model->data_type()));
  }
  if (ctx->Attr<bool>("do_bias_correction")) {
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta1_t", 0));
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta2_t", 0));
    const user_op::TensorDesc* beta1_t = ctx->TensorDesc4ArgNameAndIndex("beta1_t", 0);
    const user_op::TensorDesc* beta2_t = ctx->TensorDesc4ArgNameAndIndex("beta2_t", 0);
    JUST(CheckScalarTensorDesc(beta1_t, data_type));
    JUST(CheckScalarTensorDesc(beta2_t, data_type));
  } else {
    CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta1_t", 0));
    CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta2_t", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferIndexedSlicesAdamUpdateTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const DataType data_type = model->data_type();
  const user_op::TensorDesc* model_diff_indices =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
  const user_op::TensorDesc* model_diff_values =
      ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
  JUST(CheckIndexedSlicesModelDiffDesc(model, model_diff_indices, model_diff_values));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  if (ctx->Attr<bool>("do_bias_correction")) {
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta1_t", 0));
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("beta2_t", 0));
    const user_op::TensorDesc* beta1_t = ctx->TensorDesc4ArgNameAndIndex("beta1_t", 0);
    const user_op::TensorDesc* beta2_t = ctx->TensorDesc4ArgNameAndIndex("beta2_t", 0);
    JUST(CheckScalarTensorDesc(beta1_t, data_type));
    JUST(CheckScalarTensorDesc(beta2_t, data_type));
  } else {
    CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta1_t", 0));
    CHECK_OR_RETURN(!ctx->user_op_conf().has_input("beta2_t", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferLambUpdateTensorDesc(user_op::InferContext* ctx) {
  const float beta1 = ctx->Attr<float>("beta1");
  const float beta2 = ctx->Attr<float>("beta2");
  CHECK_GE_OR_RETURN(beta1, 0);
  CHECK_LT_OR_RETURN(beta1, 1);
  CHECK_GE_OR_RETURN(beta2, 0);
  CHECK_LT_OR_RETURN(beta2, 1);
  const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
  const DataType data_type = model->data_type();
  const Shape& shape = model->shape();
  const user_op::TensorDesc* model_diff = ctx->TensorDesc4ArgNameAndIndex("model_diff", 0);
  CHECK_EQ_OR_RETURN(model_diff->shape(), shape);
  const user_op::TensorDesc* m = ctx->TensorDesc4ArgNameAndIndex("m", 0);
  JUST(CheckTensorDescLike(m, model));
  const user_op::TensorDesc* v = ctx->TensorDesc4ArgNameAndIndex("v", 0);
  JUST(CheckTensorDescLike(v, model));
  const user_op::TensorDesc* learning_rate = ctx->TensorDesc4ArgNameAndIndex("learning_rate", 0);
  JUST(CheckLearningRateTenserDesc(learning_rate));
  const user_op::TensorDesc* beta1_t = ctx->TensorDesc4ArgNameAndIndex("beta1_t", 0);
  const user_op::TensorDesc* beta2_t = ctx->TensorDesc4ArgNameAndIndex("beta2_t", 0);
  JUST(CheckScalarTensorDesc(beta1_t, data_type));
  JUST(CheckScalarTensorDesc(beta2_t, data_type));
  if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
    const auto* scale_by_tensor = ctx->TensorDesc4ArgNameAndIndex("scale_by_tensor", 0);
    JUST(CheckScalarTensorDesc(scale_by_tensor, model->data_type()));
  }
  return Maybe<void>::Ok();
}

void SetInputArgModifierMutable(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                const std::string& arg_name, int32_t arg_index) {
  user_op::InputArgModifier* arg_modifier = GetInputArgModifierFn(arg_name, arg_index);
  CHECK_NOTNULL(arg_modifier);
  arg_modifier->set_is_mutable(true);
}

void AdamInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                          const user_op::UserOpConfWrapper& conf) {
  SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "m", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "v", 0);
  if (conf.attr<bool>("do_bias_correction")) {
    SetInputArgModifierMutable(GetInputArgModifierFn, "beta1_t", 0);
    SetInputArgModifierMutable(GetInputArgModifierFn, "beta2_t", 0);
  }
}

void LambInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                          const user_op::UserOpConfWrapper& conf) {
  SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "m", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "v", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "beta1_t", 0);
  SetInputArgModifierMutable(GetInputArgModifierFn, "beta2_t", 0);
}

REGISTER_USER_OP("sgd_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .OptionalInput("scale_by_tensor")
    .Attr<double>("scale", UserOpAttrType::kAtDouble, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn(InferSGDUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(ctx->inputs(), axis)
            .Broadcast(user_op::OpArg("learning_rate", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
    });

REGISTER_USER_OP("indexed_slices_sgd_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .SetTensorDescInferFn(InferIndexedSlicesSGDUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      const user_op::TensorDesc& model_diff_indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("learning_rate", 0))
          .Broadcast(user_op::OpArg("model_diff_indices", 0))
          .Broadcast(user_op::OpArg("model_diff_values", 0))
          .Split(user_op::OpArg("model", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("learning_rate", 0))
            .Broadcast(user_op::OpArg("model_diff_indices", 0))
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
    });

REGISTER_USER_OP("momentum_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .Input("momentum")
    .OptionalInput("scale_by_tensor")
    .Attr<double>("scale", UserOpAttrType::kAtDouble, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("beta", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn(InferMomentumUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(ctx->inputs(), axis)
            .Broadcast(user_op::OpArg("learning_rate", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
      SetInputArgModifierMutable(GetInputArgModifierFn, "momentum", 0);
    });

REGISTER_USER_OP("indexed_slices_momentum_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Input("momentum")
    .Attr<float>("beta", UserOpAttrType::kAtFloat, 0.9)
    .SetTensorDescInferFn(InferIndexedSlicesMomentumUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      const user_op::TensorDesc& model_diff_indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("learning_rate", 0))
          .Broadcast(user_op::OpArg("model_diff_indices", 0))
          .Broadcast(user_op::OpArg("model_diff_values", 0))
          .Split(user_op::OpArg("model", 0), 0)
          .Split(user_op::OpArg("momentum", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("learning_rate", 0))
            .Broadcast(user_op::OpArg("model_diff_indices", 0))
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Split(user_op::OpArg("momentum", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      SetInputArgModifierMutable(GetInputArgModifierFn, "model", 0);
      SetInputArgModifierMutable(GetInputArgModifierFn, "momentum", 0);
    });

REGISTER_USER_OP("adam_update")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .OptionalInput("scale_by_tensor")
    .Input("m")
    .Input("v")
    .OptionalInput("beta1_t")
    .OptionalInput("beta2_t")
    .Attr<double>("scale", UserOpAttrType::kAtDouble, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("beta1", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("beta2", UserOpAttrType::kAtFloat, 0.999)
    .Attr<float>("epsilon", UserOpAttrType::kAtFloat, 1e-8)
    .Attr<bool>("do_bias_correction", UserOpAttrType::kAtBool, false)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn(InferAdamUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      FOR_RANGE(int64_t, axis, 0, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(ctx->inputs())
            .Split(user_op::OpArg("model", 0), axis)
            .Split(user_op::OpArg("model_diff", 0), axis)
            .Split(user_op::OpArg("m", 0), axis)
            .Split(user_op::OpArg("v", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(AdamInputArgModifyFn);

REGISTER_USER_OP("indexed_slices_adam_update")
    .Input("model")
    .Input("model_diff_indices")
    .Input("model_diff_values")
    .Input("learning_rate")
    .Input("m")
    .Input("v")
    .OptionalInput("beta1_t")
    .OptionalInput("beta2_t")
    .Attr<float>("beta1", UserOpAttrType::kAtFloat, 0.9)
    .Attr<float>("beta2", UserOpAttrType::kAtFloat, 0.999)
    .Attr<float>("epsilon", UserOpAttrType::kAtFloat, 1e-8)
    .Attr<bool>("do_bias_correction", UserOpAttrType::kAtBool, false)
    .SetTensorDescInferFn(InferIndexedSlicesAdamUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& model = ctx->LogicalTensorDesc4InputArgNameAndIndex("model", 0);
      const user_op::TensorDesc& model_diff_indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("model_diff_indices", 0);
      std::vector<user_op::OpArg> broadcast_args;
      broadcast_args.emplace_back("learning_rate", 0);
      broadcast_args.emplace_back("model_diff_indices", 0);
      if (ctx->Attr<bool>("do_bias_correction")) {
        broadcast_args.emplace_back("beta1_t", 0);
        broadcast_args.emplace_back("beta2_t", 0);
      }
      ctx->NewBuilder()
          .Broadcast(broadcast_args)
          .Broadcast(user_op::OpArg("model_diff_values", 0))
          .Split(user_op::OpArg("model", 0), 0)
          .Split(user_op::OpArg("m", 0), 0)
          .Split(user_op::OpArg("v", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, model.shape().NumAxes()) {
        ctx->NewBuilder()
            .Broadcast(broadcast_args)
            .Split(user_op::OpArg("model_diff_values", 0),
                   model_diff_indices.shape().NumAxes() + i - 1)
            .Split(user_op::OpArg("model", 0), i)
            .Split(user_op::OpArg("m", 0), i)
            .Split(user_op::OpArg("v", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn(AdamInputArgModifyFn);

REGISTER_USER_OP("lamb_update")
    .Input("m")
    .Input("v")
    .Input("beta1_t")
    .Input("beta2_t")
    .Input("model")
    .Input("model_diff")
    .Input("learning_rate")
    .OptionalInput("scale_by_tensor")
    .Attr("beta1", UserOpAttrType::kAtFloat)
    .Attr("beta2", UserOpAttrType::kAtFloat)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .Attr<double>("scale", UserOpAttrType::kAtDouble, 1.0)
    .Attr<float>("l1", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("l2", UserOpAttrType::kAtFloat, 0.0)
    .Attr<float>("weight_decay", UserOpAttrType::kAtFloat, 0.0)
    .SetTensorDescInferFn(InferLambUpdateTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    // every bn has sbp broadcast signature
    .SetInputArgModifyFn(LambInputArgModifyFn);

}  // namespace

}  // namespace oneflow
