#include "oneflow/core/kernel/normalization_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::VirtualKernelInit(
   const ParallelContext*) {
  micro_kernel_graph_.reset(new MicroKernelGraph([&](BlobSymbolBuilder* builder){
    BlobSymbol* input = builder->NewTrainableBlobSymbol("inputs",
							"inputs_diff",
							false);
    BlobSymbol* mean = builder->NewTmpBlobSymbol("mean");
    BlobSymbol* variance = builder->NewTmpBlobSymbol("variance");
    if (JobDesc::Singleton()->IsTrain()) {
      MeanMicroKernel::Build(input, mean);
      MeanMicroKernel::Build(input, mean, variance);
    } else {
      mean = builder->NewTmpBlobSymbol("moving_mean");
      variance = builder->NewTmpBlobSymbol("moving_variance");
    }
    const auto& normalization_op_conf = this->op_conf().normalization_op();
    BlobSymbol* rsqrt = builder->NewTmpBlobSymbol("rsqrt");
    double epsilon = normalization_op_conf.epislon();
    RsqrtMicroKernel::Build(variance, epsilon, rsqrt);
    bool scale = normalization_op_conf.scale();
    bool center = normalization_op_conf.center();
    if (!scale && !center) {
      BlobSymbol* output = build->NewTmpBlobSymbol("outputs", "outputs_diff", false);
      BlobSymbol* only_fw_output = build->NewTmpBlobSymbol("outputs");
      BlobSymbol* fw_output_and_bw_input_diff =
	build->NewTmpBlobSymbol("outputs", "inputs_diff", false);
      SubMicroKernel::Build(input, mean, only_fw_output);
      MulMicroKernel::Build(fw_output_and_bw_input_diff, rsqrt, output);
    } else {
      TODO();
    }
  }));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  micro_kernel_graph_->Forward<device_type, T>(ctx, BnInOp2Blob);
  UpdateMovingMeanAndVariance();
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  micro_kernel_graph_->Backward<device_type, T>(ctx, BnInOp2Blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf,
                           NormalizationKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
