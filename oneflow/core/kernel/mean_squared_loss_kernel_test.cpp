#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename PredType>
struct MeanSquaredLossTestUtil final {
#define MEAN_SQUARED_LOSS_TEST_UTIL_ENTRY(func_name, T) \
  MeanSquaredLossTestUtil<device_type, PredType>::template func_name<T>

  DEFINE_STATIC_SWITCH_FUNC(
      void, Test, MEAN_SQUARED_LOSS_TEST_UTIL_ENTRY,
      MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));

  template<typename LabelType>
  static void Test(OpKernelTestCase* test_case, const std::string& job_type,
                   const std::string& fw_or_bw) {
    test_case->set_is_train(job_type == "train");
    test_case->set_is_forward(fw_or_bw == "forward");
    test_case->mut_op_conf()->mutable_mean_squared_loss_conf();
    BlobDesc* label_blob_desc = new BlobDesc(
        Shape({5, 2}), GetDataType<LabelType>::value, false, false, 1);
    BlobDesc* pred_blob_desc = new BlobDesc(
        Shape({5, 2}), GetDataType<PredType>::value, false, false, 1);
    BlobDesc* loss_blob_desc =
        new BlobDesc(Shape({5}), GetDataType<PredType>::value, false, false, 1);
    test_case->InitBlob<LabelType>("label", label_blob_desc,
                                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    test_case->InitBlob<PredType>("prediction", pred_blob_desc,
                                  {2, 4, 3, 7, 6, 5, 1, 9, 10, 8});
    test_case->ForwardCheckBlob<PredType>("loss", loss_blob_desc,
                                          {1.25, 2.25, 0.50, 9.25, 1.25});
    test_case->BackwardCheckBlob<PredType>(
        GenDiffBn("prediction"), pred_blob_desc,
        {0.5, 1.0, 0, 1.5, 0.5, -0.5, -3, 0.5, 0.5, -1});
  }
};

template<DeviceType device_type, typename PredType>
void MeanSquaredLossKernelTestCase(OpKernelTestCase* test_case,
                                   const std::string& label_type,
                                   const std::string& job_type,
                                   const std::string& fw_or_bw) {
  MeanSquaredLossTestUtil<device_type, PredType>::SwitchTest(
      SwitchCase(label_type), test_case, job_type, fw_or_bw);
}

TEST_CPU_AND_GPU_OPKERNEL(MeanSquaredLossKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, INT_DATA_TYPE_SEQ),
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
