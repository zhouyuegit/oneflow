#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
class ConvFilterGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradGpuKernel);
  ConvFilterGradGpuKernel() = default;
  ~ConvFilterGradGpuKernel() = default;

 private:
  const PbMessage &GetCustomizedOpConf() const override {
    return this->op_conf().conv_filter_grad_conf();
  }

  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    CudnnConvArgs args(this->op_conf().conv_filter_grad_conf().conv_conf(),
                       ctx.device_ctx->cudnn_handle(), BnInOp2Blob("x"), BnInOp2Blob("dy"),
                       BnInOp2Blob("filter_diff"), BnInOp2Blob("buf"),
                       this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                       this->job_desc().job_conf().cudnn_conv_heuristic_search_algo());
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t work_space_size = 0;
    if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_filter_algo()) {
      algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->job_desc().job_conf().cudnn_conv_force_bwd_filter_algo());
      CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    } else {
      algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->kernel_conf().conv_filter_grad_conf().cudnn_bwd_filter_algo());
      CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    }
    LOG(INFO) << "cudnn conv filter grad @ " << this->op_conf().name();
    LOG(INFO) << "cudnn conv filter grad @ x static shape: "
              << BnInOp2Blob("x")->static_shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ y static shape: "
              << BnInOp2Blob("dy")->static_shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ filter static shape: "
              << BnInOp2Blob("filter_diff")->static_shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ x dynamic shape: "
              << BnInOp2Blob("x")->shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ y dynamic shape: "
              << BnInOp2Blob("dy")->shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ filter dynamic shape: "
              << BnInOp2Blob("filter_diff")->shape().ToString();
    LOG(INFO) << "cudnn conv filter grad @ algo: " << algo;
    LOG(INFO) << "cudnn conv filter grad @ algo needed buffer size: " << work_space_size;
    LOG(INFO) << "cudnn conv filter grad @ buf blob bytes size: "
              << BnInOp2Blob("buf")->ByteSizeOfBlobBody();
    CHECK_LE(work_space_size, BnInOp2Blob("buf")->ByteSizeOfBlobBody());
    CudaCheck(cudnnConvolutionBackwardFilter(
        args.handle, CudnnSPOnePtr<T>(), args.xdesc.Get(), args.x_dptr, args.ydesc.Get(),
        args.y_dptr, args.cdesc.Get(), algo, args.work_space, work_space_size, CudnnSPZeroPtr<T>(),
        args.wdesc.Get(), args.w_dptr));
  }
};

#define REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(dtype)                                          \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConvFilterGradConf, DeviceType::kGPU, \
                                        dtype, ConvFilterGradGpuKernel<dtype>);

REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(double);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float16);

}  // namespace oneflow
