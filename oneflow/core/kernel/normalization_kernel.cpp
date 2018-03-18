#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf inv_elem_num_init_conf;
  float elem_cnt = BnInOp2Blob("in")->shape().elem_cnt();
  inv_elem_num_init_conf.mutable_constant_conf()->set_value(1.0 / elem_cnt);
  KernelUtil<device_type, T>::Initialize(ctx, inv_elem_num_init_conf, 0,
                                         BnInOp2Blob("inv_elem_cnt"));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithOpConf(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().normalization_conf().scale()) {
    InitializerConf gamma_init_conf;
    float gamma_init = this->op_conf().normalization_conf().gamma_init();
    gamma_init_conf.mutable_constant_conf()->set_value(gamma_init);
    KernelUtil<device_type, T>::Initialize(ctx, gamma_init_conf, 0,
                                           BnInOp2Blob("gamma"));
  }
  if (this->op_conf().normalization_conf().center()) {
    InitializerConf beta_init_conf;
    float beta_init = this->op_conf().normalization_conf().beta_init();
    beta_init_conf.mutable_constant_conf()->set_value(beta_init);
    KernelUtil<device_type, T>::Initialize(ctx, beta_init_conf, 0,
                                           BnInOp2Blob("beta"));
  }
  InitializerConf moving_mean_init_conf;
  moving_mean_init_conf.mutable_constant_conf()->set_value(0.f);
  KernelUtil<device_type, T>::Initialize(ctx, moving_mean_init_conf, 0,
                                         BnInOp2Blob("moving_mean"));
  InitializerConf moving_variance_init_conf;
  moving_variance_init_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, moving_variance_init_conf, 0,
                                         BnInOp2Blob("moving_variance"));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().normalization_conf().scale()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, gamma_blob, "gamma", 1, 1);
  }

  if (this->op_conf().normalization_conf().center()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, beta_blob, "beta", 1, 1);
  }

  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, moving_mean_blob, "moving_mean",
      1, 1);

  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, moving_variance_blob,
      "moving_variance", 1, 1);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* mean_blob = nullptr;
  const Blob* variance_blob = nullptr;
  if (JobDesc::Singleton()->IsTrain()) {
    CalcMeanAndVariance(ctx, BnInOp2Blob);
    UpdateMovingMeanAndMovingVariance(ctx, BnInOp2Blob);
    mean_blob = BnInOp2Blob("mean");
    variance_blob = BnInOp2Blob("variance");
  } else {
    mean_blob = BnInOp2Blob("moving_mean");
    variance_blob = BnInOp2Blob("moving_variance");
  }
  Normalize(ctx, BnInOp2Blob, mean_blob, variance_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const Blob* out_diff = BnInOp2Blob("out_diff");
  if (normalization_op_conf.center()) {
    Blob* beta_diff_blob = BnInOp2Blob("beta_diff");
    Blob* tmp_storage_blob = BnInOp2Blob("tmp_storage_for_sum");
    KernelUtil<device_type, T>::Sum(
        ctx.device_ctx, out_diff->shape().elem_cnt(), out_diff->dptr<T>(),
        beta_diff_blob->mut_dptr<T>(), tmp_storage_blob->mut_dptr<T>(),
        tmp_storage_blob->shape().elem_cnt());
  }

  Blob* in_diff = BnInOp2Blob("in_diff");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  if (normalization_op_conf.scale()) {
    Blob* gamma_diff_blob = BnInOp2Blob("gamma_diff");
    const Blob* normalized_in_blob = BnInOp2Blob("normalized_in");
    KernelUtil<device_type, T>::Dot(
        ctx.device_ctx, out_diff->shape().elem_cnt(), out_diff->dptr<T>(), 1,
        normalized_in_blob->dptr<T>(), 1, gamma_diff_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Scal(ctx.device_ctx, 1,
                                     BnInOp2Blob("gamma")->dptr<T>(),
                                     inv_var_blob->mut_dptr<T>(), 1);
  }
  NormalizationKernelUtil<device_type, T>::Scal(
      ctx.device_ctx, out_diff->shape().elem_cnt(), out_diff->dptr<T>(),
      inv_var_blob->dptr<T>(), in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::Normalize(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* mean_blob, const Blob* variance_blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  NormalizationKernelUtil<device_type, T>::Rsqrt(
      ctx.device_ctx, 1, variance_blob->dptr<T>(),
      normalization_op_conf.epsilon(), inv_var_blob->mut_dptr<T>());
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  if (!normalization_op_conf.scale() && !normalization_op_conf.center()) {
    normalized_blob = BnInOp2Blob("out");
  }
  NormalizationKernelUtil<device_type, T>::ScalarSub(
      ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
      mean_blob->dptr<T>(), normalized_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Scal(
      ctx.device_ctx, normalized_blob->shape().elem_cnt(),
      inv_var_blob->dptr<T>(), normalized_blob->mut_dptr<T>(), 1);
  if (normalization_op_conf.scale() || normalization_op_conf.center()) {
    Blob* out_blob = BnInOp2Blob("out");
    Blob* gamma_scaled_blob = nullptr;
    if (normalization_op_conf.scale()) {
      gamma_scaled_blob = out_blob;
      const Blob* gamma_blob = BnInOp2Blob("gamma");
      NormalizationKernelUtil<device_type, T>::Scal(
          ctx.device_ctx, normalized_blob->shape().elem_cnt(),
          normalized_blob->dptr<T>(), gamma_blob->dptr<T>(),
          gamma_scaled_blob->mut_dptr<T>());
    } else {
      gamma_scaled_blob = normalized_blob;
    }

    if (normalization_op_conf.center()) {
      const Blob* beta_blob = BnInOp2Blob("beta");
      NormalizationKernelUtil<device_type, T>::ScalarAdd(
          ctx.device_ctx, gamma_scaled_blob->shape().elem_cnt(),
          gamma_scaled_blob->dptr<T>(), beta_blob->dptr<T>(),
          out_blob->mut_dptr<T>());
    }
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcMeanAndVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* mean_blob = BnInOp2Blob("mean");
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* tmp_storage_blob = BnInOp2Blob("tmp_storage_for_sum");
  KernelUtil<device_type, T>::Sum(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                  in_blob->dptr<T>(), mean_blob->mut_dptr<T>(),
                                  tmp_storage_blob->mut_dptr<T>(),
                                  tmp_storage_blob->shape().elem_cnt());
  const Blob* inv_elem_num_blob = BnInOp2Blob("inv_elem_num");
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, 1,
                                   inv_elem_num_blob->dptr<T>(),
                                   mean_blob->mut_dptr<T>(), 1);

  //  It's safe to use `out' as tmp blob
  Blob* tmp_blob = BnInOp2Blob("out");
  NormalizationKernelUtil<device_type, T>::ScalarSub(
      ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
      mean_blob->dptr<T>(), tmp_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, tmp_blob->shape().elem_cnt(),
                                  tmp_blob->dptr<T>(), tmp_blob->dptr<T>(),
                                  tmp_blob->mut_dptr<T>());
  Blob* variance_blob = BnInOp2Blob("variance");
  KernelUtil<device_type, T>::Sum(
      ctx.device_ctx, tmp_blob->shape().elem_cnt(), tmp_blob->dptr<T>(),
      variance_blob->mut_dptr<T>(), tmp_storage_blob->mut_dptr<T>(),
      tmp_storage_blob->shape().elem_cnt());
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, 1,
                                   inv_elem_num_blob->dptr<T>(),
                                   variance_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::UpdateMovingMeanAndMovingVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  T momentum = this->op_conf().normalization_conf().momentum();
  const Blob* mean_blob = BnInOp2Blob("mean");
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, mean_blob->shape().elem_cnt(), momentum,
      mean_blob->dptr<T>(), 1, moving_mean_blob->mut_dptr<T>(), 1);
  const Blob* variance_blob = BnInOp2Blob("variance");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, variance_blob->shape().elem_cnt(), momentum,
      variance_blob->dptr<T>(), 1, moving_variance_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf,
                           NormalizationKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
