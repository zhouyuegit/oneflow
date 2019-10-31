#include "oneflow/core/operator/operator.h"

namespace oneflow {
class MaskerOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskerOp);
  MaskerOp() = default;
  ~MaskerOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_masker_conf());
    EnrollInputBn("mask_prob", false);
    EnrollInputBn("box", false);
    EnrollInputBn("image_size", false);
    EnrollTmpBn("mask_prob_buf");
    EnrollTmpBn("box_buf");
    EnrollTmpBn("mask_buf");
    EnrollOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().masker_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const MaskerOpConf& conf = op_conf().masker_conf();
    // input: mask_prob (R, M_H, M_W)
    const BlobDesc* mask_prob = GetBlobDesc4BnInOp("mask_prob");
    CHECK_EQ_OR_RETURN(mask_prob->shape().NumAxes(), 3);
    CHECK_OR_RETURN(IsFloatingDataType(mask_prob->data_type()));
    const int32_t num_mask = mask_prob->shape().At(0);
    // input: box (R, 4)
    const BlobDesc* box = GetBlobDesc4BnInOp("box");
    CHECK_EQ_OR_RETURN(box->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(box->shape().At(1), 4);
    CHECK_EQ_OR_RETURN(num_mask, box->shape().At(0));
    // Use the same data type for box and mask_prob to reduce template param
    CHECK_EQ_OR_RETURN(box->data_type(), mask_prob->data_type());
    // input: image_size(2,), only used to infer output's dense_shape
    const BlobDesc* image_size = GetBlobDesc4BnInOp("image_size");
    CHECK_EQ_OR_RETURN(image_size->shape().NumAxes(), 1);
    CHECK_EQ_OR_RETURN(image_size->shape().elem_cnt(), 2);
    CHECK_EQ_OR_RETURN(image_size->data_type(), DataType::kInt32);
    CHECK_OR_RETURN(!image_size->is_dynamic());

    // tmp: mask prob buffer (M_H + 2 * padding, M_W + 2 * padding)
    BlobDesc* mask_prob_buf = GetBlobDesc4BnInOp("mask_prob_buf");
    mask_prob_buf->mut_shape() = Shape({mask_prob->shape().At(1) + 2 * conf.padding(),
                                        mask_prob->shape().At(2) + 2 * conf.padding()});
    mask_prob_buf->set_data_type(mask_prob->data_type());
    mask_prob_buf->set_is_dynamic(mask_prob->is_dynamic());
    // tmp: box buffer (4,)
    BlobDesc* box_buf = GetBlobDesc4BnInOp("box_buf");
    box_buf->mut_shape() = Shape({4});
    box_buf->set_data_type(box->data_type());
    // tmp: mask buffer (max_image_height, max_image_width)
    BlobDesc* mask_buf = GetBlobDesc4BnInOp("mask_buf");
    mask_buf->mut_shape() = Shape({conf.max_image_height(), conf.max_image_width()});
    mask_buf->set_data_type(DataType::kInt8);
    mask_buf->set_is_dynamic(true);

    // output: (R, max_image_height, max_image_width)
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->mut_shape() = Shape({num_mask, conf.max_image_height(), conf.max_image_width()});
    out->set_data_type(DataType::kInt8);
    // out is set to be dynamic for 2 reasons:
    // 1) can't judge whether out is dynamic by inputs
    // 2) out is always dynamic in MaskRCNN for current implementation
    out->set_is_dynamic(true);

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    const OptInt64* mask_prob_batch_axis = BatchAxis4BnInOp("mask_prob");
    const OptInt64* box_batch_axis = BatchAxis4BnInOp("box");
    CHECK_OR_RETURN(mask_prob_batch_axis->has_value());
    CHECK_OR_RETURN(box_batch_axis->has_value());
    CHECK_EQ_OR_RETURN(box_batch_axis->value(), mask_prob_batch_axis->value());
    *BatchAxis4BnInOp("out") = *mask_prob_batch_axis;
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("mask_prob")->data_type());
  }
};

REGISTER_CPU_OP(OperatorConf::kMaskerConf, MaskerOp);

}  // namespace oneflow
