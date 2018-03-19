#include "oneflow/core/kernel/normalization_model_update_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationMdupdtConf,
                           NormalizationModelUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
