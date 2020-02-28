#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("count_nonzero")
    .Input("in")
    .Output("out")
    .Input("axis", UserOpAttrType::kAtInt64)
    .Attr("keepdims", UserOpAttrType::kAtBool)
    .Attr("dtype", UserOpAttrType::kAtInt64)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *in_shape = *out_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("count_nonzero_grad")
    .Input("out_diff")
    .Input("dx")
    .Input("axis", UserOpAttrType::kAtInt64)
    .Output("in_diff")
    .Attr("keepdims", UserOpAttrType::kAtBool)
    .Attr("dtype", UserOpAttrType::kAtInt64)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_diff_shape = ctx->Shape4ArgNameAndIndex("out_diff", 0);
      Shape* in_diff_shape = ctx->Shape4ArgNameAndIndex("in_diff", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dx_shape == *out_diff_shape);
      *dx_shape = *out_diff_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in_diff", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("count_nonzero").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper count_nonzero_grad_op =
        builder.Op("unary_grad")
            .Input("out_diff", op.input("in", 0))
            .Input("dx", op.GetGradTensorWithOpOutput("out", 0))
            .Input("axis", op.GetGradTensorWithOpOutput("axis", 0))
            .Output("in_diff")
            .Attr("keepdims", true)
            .Attr<int64_t>("dtype", 0)
            .Build();
    op.BindGradTensorWithOpInput(count_nonzero_grad_op.output("dx", 0), "out_diff", 0);
    AddOp(count_nonzero_grad_op);
  }
});

}  // namespace oneflow
