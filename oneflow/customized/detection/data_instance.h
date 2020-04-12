#ifndef ONEFLOW_CUSTOMIZED_DETECTION_DATA_INSTANCE_H_
#define ONEFLOW_CUSTOMIZED_DETECTION_DATA_INSTANCE_H_

#include "oneflow/customized/detection/data_field.h"

namespace oneflow {

namespace detection {

class DataInstance {
 public:
  DataInstance() = default;
  void InitFromProto(const DataInstanceProto& proto);

  template<DetectionDataCase dsrc, typename... Args>
  DataField* GetOrCreateField(Args&&... args);

  template<DetectionDataCase dsrc>
  DataField* GetField();
  DataField* GetField(DetectionDataCase dsrc);
  template<DetectionDataCase dsrc>
  const DataField* GetField() const;
  const DataField* GetField(DetectionDataCase dsrc) const;
  template<DetectionDataCase dsrc>
  bool HasField() const;
  bool AddField(std::unique_ptr<DataField>&& data_field_ptr);
  void Transform(const DataTransformProto& trans_proto);

 private:
  HashMap<DetectionDataCase, std::unique_ptr<DataField>, std::hash<int>> fields_;
};

inline const DataField* DataInstance::GetField(DetectionDataCase dsrc) const {
  if (fields_.find(dsrc) == fields_.end()) { return nullptr; }
  return const_cast<const DataField*>(fields_.at(dsrc).get());
}

inline DataField* DataInstance::GetField(DetectionDataCase dsrc) {
  if (fields_.find(dsrc) == fields_.end()) { return nullptr; }
  return fields_.at(dsrc).get();
}

template<DetectionDataCase dsrc>
inline DataField* DataInstance::GetField() {
  return GetField(dsrc);
}

template<DetectionDataCase dsrc>
inline const DataField* DataInstance::GetField() const {
  return GetField(dsrc);
}

template<DetectionDataCase dsrc>
inline bool DataInstance::HasField() const {
  return fields_.find(dsrc) != fields_.end();
}

inline bool DataInstance::AddField(std::unique_ptr<DataField>&& data_field_ptr) {
  return fields_.emplace(data_field_ptr->data_case(), std::move(data_field_ptr)).second;
}

template<DetectionDataCase dsrc, typename... Args>
DataField* DataInstance::GetOrCreateField(Args&&... args) {
  if (fields_.find(dsrc) == fields_.end()) {
    using DataFieldT = typename DataFieldTrait<dsrc>::type;
    std::unique_ptr<DataField> data_field_ptr;
    data_field_ptr.reset(new DataFieldT(std::forward<Args>(args)...));
    data_field_ptr->set_data_case(dsrc);
    AddField(std::move(data_field_ptr));
  }
  return fields_.at(dsrc).get();
}

}  // namespace detection

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DETECTION_DATA_INSTANCE_H_
