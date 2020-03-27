#include "oneflow/customized/detection/data_instance.h"
#include "oneflow/customized/detection/data_transform.h"

namespace oneflow {

namespace detection {

void DataInstance::InitFromProto(const DataInstanceProto& proto) {
  for (const auto& field_proto : proto.data_fields()) {
    auto data_field_ptr = CreateDataFieldFromProto(field_proto);
    CHECK(AddField(std::move(data_field_ptr)));
  }
}

void DataInstance::Transform(const DataTransformProto& trans_proto) {
  DataTransform(this, trans_proto);
}

}  // namespace detection

}  // namespace oneflow
