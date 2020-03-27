#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_TRANSFORM_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_TRANSFORM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/customized/detection/data_instance.h"

namespace oneflow {

namespace detection {

void DataTransform(DataInstance* data_inst, const DataTransformProto& trans_proto);
void BatchTransform(std::shared_ptr<std::vector<DataInstance>> batch_data_inst_ptr,
                    const DataTransformProto& trans_proto);

using TransformCase = DataTransformProto::TransformCase;

template<TransformCase trans>
void DoDataTransform(DataInstance* data_inst, const DataTransformProto& proto);

template<TransformCase trans>
void DoBatchTransform(std::shared_ptr<std::vector<DataInstance>> batch_data_inst_ptr,
                      const DataTransformProto& proto);

template<DataSourceCase dsrc, TransformCase trans>
struct DataTransformer;

#define DATA_TRANSFORM_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kResize) \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kTargetResize)

#define DATA_FIELD_TRANSFORM_TUPLE_SEQ                                    \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, DATA_SOURCE_SEQ, \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, DATA_TRANSFORM_SEQ)

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_TRANSFORM_H_
