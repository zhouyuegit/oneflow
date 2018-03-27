#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* NormalizationTestCase(bool is_train, bool is_forward) {
  OpKernelTestCase* norm_test_case = new OpKernelTestCase();
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(is_forward);
  norm_test_case->set_device_type(device_type);
  auto conf = norm_test_case->mut_op_conf()->mutable_relu_conf();

  using KTC = KTCommon<device_type, T>;
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::val, false, false, 1);
  norm_test_case->InitBlob("inputs",
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));
  norm_test_case->InitBlob("moving_mean",
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {0}));
  norm_test_case->InitBlob("moving_variance",
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));
  norm_test_case->InitBlob("beta",
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {0}));
  norm_test_case->InitBlob("gamma",
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));
  norm_test_case->InitBlob(GenDiffBn("outputs"),
                           KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));

  norm_test_case->ForwardCheckBlob(
      "outputs", device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {31.622776}));
  /*
  norm_test_case->ForwardCheckBlob(
      "moving_mean", device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {0}));
  norm_test_case->ForwardCheckBlob(
      "moving_variance", device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {0.99}));
      */

  norm_test_case->BackwardCheckBlob(
      GenDiffBn("inputs"), device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));
  norm_test_case->BackwardCheckBlob(
      GenDiffBn("beta"), device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1}));
  norm_test_case->BackwardCheckBlob(
      GenDiffBn("gamma"), device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {31.622776}));

  return norm_test_case;
}

#define MAKE_ENTRY(device_type, data_type_pair, is_train, is_forward) \
  TEST(NormalizationKernel,                                           \
       OF_PP_JOIN(normalization_, __COUNTER__, _, device_type, _,     \
                  OF_PP_PAIR_FIRST(data_type_pair), _, is_train, _,   \
                  is_forward)) {                                      \
    NormalizationTestCase<DeviceType::k##device_type,                 \
                          OF_PP_PAIR_FIRST(data_type_pair)>(          \
        std::string(#is_train) == "train",                            \
        std::string(#is_forward) == "forward")                        \
        ->Run();                                                      \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, (CPU), FLOATING_DATA_TYPE_SEQ,
                                 (train), (forward)(backward))

}  // namespace test

}  // namespace oneflow
