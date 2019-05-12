#ifndef ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_CUH_
#define ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_CUH_

namespace oneflow {

template<typename T>
__host__ __device__ T RegDiff(const T diff, const T batch_instance_num, const T l1, const T l2,
                              const T pre_model_val) {
  return diff / batch_instance_num + l1 * ((pre_model_val >= 0) - (pre_model_val <= 0))
         + l2 * pre_model_val;
}

template<typename T>
__host__ __device__ T NaiveUpdateModel(const T learning_rate, const T corrected_direction,
                                       const T weight_decay, const T pre_model_val) {
  return pre_model_val - learning_rate * (corrected_direction + weight_decay * pre_model_val);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_CUH_
