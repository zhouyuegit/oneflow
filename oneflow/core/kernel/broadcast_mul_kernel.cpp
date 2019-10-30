#include "oneflow/core/kernel/broadcast_mul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

bool CanCompress(const int64_t pre_a, const int64_t pre_b, const int64_t a, const int64_t b) {
  return (pre_a == pre_b && a == b) || (pre_a == 1 && a == 1) || (pre_b == 1 && b == 1);
}

void CreateCompressedShape(const Shape& a_shape, const Shape& b_shape, Shape* compressed_a_shape,
                           Shape* compressed_b_shape, Shape* compressed_out_shape) {
  const int64_t num_extended_axes = std::max(a_shape.NumAxes(), b_shape.NumAxes());
  const Shape extended_a_shape = a_shape.CreateLeftExtendedShape(num_extended_axes);
  const Shape extended_b_shape = b_shape.CreateLeftExtendedShape(num_extended_axes);
  std::vector<int64_t> compressed_a_dim_vec;
  std::vector<int64_t> compressed_b_dim_vec;
  FOR_RANGE(int64_t, i, 0, num_extended_axes) {
    if (i != 0
        && CanCompress(compressed_a_dim_vec.back(), compressed_b_dim_vec.back(),
                       extended_a_shape.At(i), extended_b_shape.At(i))) {
      compressed_a_dim_vec.back() *= extended_a_shape.At(i);
      compressed_b_dim_vec.back() *= extended_b_shape.At(i);
    } else {
      compressed_a_dim_vec.push_back(extended_a_shape.At(i));
      compressed_b_dim_vec.push_back(extended_b_shape.At(i));
    }
  }
  *compressed_a_shape = Shape(compressed_a_dim_vec);
  *compressed_b_shape = Shape(compressed_b_dim_vec);
  std::vector<int64_t> compressed_out_dim_vec(compressed_a_dim_vec.size());
  std::transform(compressed_a_dim_vec.cbegin(), compressed_a_dim_vec.cend(),
                 compressed_b_dim_vec.cbegin(), compressed_out_dim_vec.begin(),
                 [](const int64_t a, const int64_t b) { return std::max(a, b); });
  *compressed_out_shape = Shape(compressed_out_dim_vec);
}

}  // namespace

template<DeviceType device_type, typename T>
void BroadcastMulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  Blob* out = BnInOp2Blob("out");
  int64_t n = out->shape().elem_cnt();
  if (a->shape().elem_cnt() == 1) {
    CHECK_EQ(n, b->shape().elem_cnt());
    KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, b->dptr<T>(), a->dptr<T>(),
                                            out->mut_dptr<T>());
  } else if (b->shape().elem_cnt() == 1) {
    CHECK_EQ(n, a->shape().elem_cnt());
    KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, a->dptr<T>(), b->dptr<T>(),
                                            out->mut_dptr<T>());
  } else {
    Shape compressed_a_shape;
    Shape compressed_b_shape;
    Shape compressed_out_shape;
    CreateCompressedShape(a->shape(), b->shape(), &compressed_a_shape, &compressed_b_shape,
                          &compressed_out_shape);
    CHECK_EQ(compressed_a_shape.elem_cnt(), a->shape().elem_cnt());
    CHECK_EQ(compressed_b_shape.elem_cnt(), b->shape().elem_cnt());
    CHECK_EQ(compressed_out_shape.elem_cnt(), out->shape().elem_cnt());
    if (compressed_a_shape.NumAxes() == 2
        && (compressed_a_shape.At(0) == compressed_b_shape.At(0)
            || compressed_a_shape.At(1) == compressed_b_shape.At(1))) {
      if (compressed_a_shape.At(0) == compressed_b_shape.At(0)) {
        if (compressed_a_shape.At(1) == 1) {
          KernelUtil<device_type, T>::MulByCol(ctx.device_ctx, compressed_b_shape.At(0),
                                               compressed_b_shape.At(1), b->dptr<T>(), a->dptr<T>(),
                                               out->mut_dptr<T>());
        } else if (compressed_b_shape.At(1) == 1) {
          KernelUtil<device_type, T>::MulByCol(ctx.device_ctx, compressed_a_shape.At(0),
                                               compressed_a_shape.At(1), a->dptr<T>(), b->dptr<T>(),
                                               out->mut_dptr<T>());
        } else {
          UNIMPLEMENTED();
        }
      } else {
        UNIMPLEMENTED();
      }
    } else {
      NdarrayUtil<device_type, T>::BroadcastMul(
          ctx.device_ctx, XpuVarNdarray<T>(compressed_out_shape, out->mut_dptr<T>()),
          XpuVarNdarray<const T>(compressed_a_shape, a->dptr<T>()),
          XpuVarNdarray<const T>(compressed_b_shape, b->dptr<T>()));
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMulConf, BroadcastMulKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
