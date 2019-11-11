#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

namespace {

struct DeviceDimSliceConf {
  DeviceDimSliceConf(const int32_t start_, const int32_t end_, const int32_t stride_)
      : start(start_), end(end_), stride(stride_) {}
  OF_DEVICE_FUNC DeviceDimSliceConf() = default;
  OF_DEVICE_FUNC DeviceDimSliceConf(const DeviceDimSliceConf&) = default;

  int32_t start;
  int32_t end;
  int32_t stride;
};

struct DeviceSliceConf {
  explicit DeviceSliceConf(const SliceOpConf& conf) : num_axes(conf.dim_slice_conf_size()) {
    FOR_RANGE(int32_t, i, 0, OF_PP_SEQ_SIZE(DIM_SEQ)) {
      if (i < num_axes) {
        const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(i);
        dim_conf[i] = DeviceDimSliceConf(dim_slice_conf.start(), dim_slice_conf.end(),
                                         dim_slice_conf.stride());
      } else {
        dim_conf[i] = DeviceDimSliceConf(0, 0, 0);
      }
    }
  }
  OF_DEVICE_FUNC DeviceSliceConf(const DeviceSliceConf&) = default;

  DeviceDimSliceConf dim_conf[OF_PP_SEQ_SIZE(DIM_SEQ)];
  int32_t num_axes;
};

template<typename T, size_t NDIMS>
__global__ void SliceForwardGpu(const int64_t n, XpuShape in_shape, const T* in_ptr,
                                DeviceSliceConf slice_conf, XpuShape out_shape, T* out_ptr) {
  int64_t in_coord[NDIMS];
  int64_t out_coord[NDIMS];
  CUDA_1D_KERNEL_LOOP(i, n) {
    XpuShapeUtil<NDIMS>::Offset2Coordinate(out_shape, i, out_coord[3]);
#pragma unroll
    for (int64_t j = 0; j < NDIMS; ++j) {
      const DeviceDimSliceConf& dim_slice_conf = slice_conf.dim_conf[j];
      in_coord[j] = dim_slice_conf.start + out_coord[j] * dim_slice_conf.stride;
    }
    out_ptr[i] = in_ptr[XpuShapeUtil<NDIMS>::Coordinate2Offset(in_shape, in_coord[NDIMS])];
  }
}

template<typename T>
__global__ void SliceBackwardGpu(const int64_t n, const int64_t* offset, const T* slice,
                                 T* entire) {
  // CUDA_1D_KERNEL_LOOP(i, n) { entire[offset[i]] = slice[i]; }
}

}  // namespace

template<typename T>
class SliceGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGpuKernel);
  SliceGpuKernel() = default;
  ~SliceGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    const DeviceSliceConf slice_conf(this->op_conf().slice_conf());
    const int64_t num_output = out_blob->shape().elem_cnt();
    SliceForwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        num_output, XpuShape(in_blob->shape()), in_blob->dptr<T>(), slice_conf,
        XpuShape(out_blob->shape()), out_blob->mut_dptr<T>());
  }
};

template<typename T>
class SliceGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradGpuKernel);
  SliceGradGpuKernel() = default;
  ~SliceGradGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* like_blob = BnInOp2Blob("like");
    Blob* offset_blob = BnInOp2Blob("y_to_x_offset");
    Blob* dx_blob = BnInOp2Blob("dx");

    // TODO: Add InferTmpBlobDenseShape for op with dynamic tmp blob
    offset_blob->dense_shape_mut_view().set_shape(static_cast<Shape>(dy_blob->shape()));

    const SliceGradOpConf& conf = op_conf().slice_grad_conf();
    WithHostBlobAndStreamSynchronizeEnv(ctx.device_ctx, offset_blob, [&](Blob* host_blob) {
      int64_t* host_blob_ptr = host_blob->mut_dptr<int64_t>();
      FOR_RANGE(int64_t, i, 0, host_blob->shape().elem_cnt()) {
        int64_t offset = 0;
        int64_t index = i;
        FOR_RANGE(int64_t, j, 0, host_blob->shape().NumAxes()) {
          const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(j);
          const int64_t dim_len = like_blob->shape().At(j);
          const int64_t dim_elem_cnt = host_blob->shape().Count(j + 1);
          index = index % dim_elem_cnt;
          const int64_t dim_i = index / dim_elem_cnt;
          int64_t start = dim_slice_conf.has_start() ? dim_slice_conf.start() : 0;
          if (start < 0) { start += dim_len; }
          CHECK_GE(start, 0);
          CHECK_LT(start, dim_len);
          int64_t stride = dim_slice_conf.stride();
          CHECK_GT(stride, 0);
          offset += (start + dim_i * stride) * like_blob->shape().Count(j + 1);
        }
        host_blob_ptr[i] = offset;
      }
    });
    Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                             dx_blob->ByteSizeOfBlobBody());
    const int64_t num_output = dy_blob->shape().elem_cnt();
    SliceBackwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
        num_output, offset_blob->dptr<int64_t>(), dy_blob->dptr<T>(), dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_SLICE_GPU_KERNEL(dtype)                                                       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceConf, DeviceType::kGPU, dtype,     \
                                        SliceGpuKernel<dtype>)                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceGradConf, DeviceType::kGPU, dtype, \
                                        SliceGradGpuKernel<dtype>)

REGISTER_SLICE_GPU_KERNEL(float);
REGISTER_SLICE_GPU_KERNEL(double);
REGISTER_SLICE_GPU_KERNEL(int8_t);
REGISTER_SLICE_GPU_KERNEL(int32_t);
REGISTER_SLICE_GPU_KERNEL(int64_t);

}  // namespace oneflow
