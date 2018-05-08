#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/mean_squared_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void MeanSquaredLossCopyLabel2DiffGpu(const int64_t elem_cnt, const LabelType* label,
                                                 PredType* diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { diff[i] = static_cast<PredType>(label[i]); }
}

}  // namespace

template<typename PredType, typename LabelType>
struct MeanSquaredLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t inst_num, const int64_t label_dim,
                      const LabelType* label, const PredType* pred, PredType* diff,
                      PredType* loss) {
    const int64_t n = inst_num * label_dim;
    MeanSquaredLossCopyLabel2DiffGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                                       ctx->cuda_stream()>>>(n, label, diff);
    KernelUtil<DeviceType::kGPU, PredType>::Axpy(ctx, n, static_cast<PredType>(-1), pred, 1, diff,
                                                 1);
    for (int64_t i = 0; i < inst_num; ++i) {
      KernelUtil<DeviceType::kGPU, PredType>::Dot(ctx, label_dim, diff + i * label_dim, 1,
                                                  diff + i * label_dim, 1, loss + i);
    }
    KernelUtil<DeviceType::kGPU, PredType>::Div(ctx, inst_num, loss,
                                                static_cast<PredType>(2 * label_dim));
  }

  static void Backward(DeviceCtx* ctx, const int64_t inst_num, const int64_t label_dim,
                       const PredType* diff, PredType* pred_diff) {
    const int64_t n = inst_num * label_dim;
    KernelUtil<DeviceType::kGPU, PredType>::Copy(ctx, n, diff, 1, pred_diff, 1);
    KernelUtil<DeviceType::kGPU, PredType>::Scal(ctx, n, static_cast<PredType>(-1), pred_diff, 1);
    KernelUtil<DeviceType::kGPU, PredType>::Div(ctx, n, pred_diff,
                                                static_cast<PredType>(label_dim));
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                             \
  template struct MeanSquaredLossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                            OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
