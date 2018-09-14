#include "oneflow/core/kernel/add_with_actv_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& AddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().add_with_actv_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddWithActvConf, AddWithActvKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
