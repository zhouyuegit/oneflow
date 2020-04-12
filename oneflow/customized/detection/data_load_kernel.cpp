#include "oneflow/core/kernel/kernel.h"
// #include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/detection/data_loader.h"
#include "oneflow/customized/detection/data_instance.h"

namespace oneflow {

namespace {

void InitOptVariableAxis(const DetectionDataLoadOpConf::DataBlobConf& blob_conf,
                         OptInt64* var_axis) {
  if (blob_conf.has_tensor_list_variable_axis() == false) { return; }
  var_axis->set_value(blob_conf.tensor_list_variable_axis());
}

}  // namespace

class DetectionDataLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DetectionDataLoadKernel);
  DetectionDataLoadKernel() = default;
  ~DetectionDataLoadKernel() = default;

 private:
  void VirtualKernelInit() override {
    data_loader_.reset(new detection::DataLoader(this->op_conf().detection_data_load_conf(),
                                                 this->kernel_conf().detection_data_load_conf()));
  }

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    auto batch_data = data_loader_->FetchBatch();
    FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
      Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
      const auto& blob_conf = op_conf().detection_data_load_conf().blobs(i);
      WriteDataToBlob(ctx.device_ctx, batch_data, blob_conf, out_blob);
    }
  }

  void WriteDataToBlob(DeviceCtx* ctx,
                       std::shared_ptr<std::vector<detection::DataInstance>> batch_data,
                       const DetectionDataLoadOpConf::DataBlobConf& blob_conf, Blob* blob) const {
    using namespace detection;
    CHECK(blob_conf.has_tensor_list_variable_axis() == blob->blob_desc().is_tensor_list());
    OptInt64 var_axis;
    InitOptVariableAxis(blob_conf, &var_axis);
    char* dptr = static_cast<char*>(blob->mut_dptr());
    Memset<DeviceType::kCPU>(ctx, dptr, 0, blob->AlignedByteSizeOfBlobBody());
    if (var_axis.has_value() == false) {
      Shape instance_shape;
      const DataField* first = batch_data->at(0).GetField(blob_conf.data_case());
      first->InferShape(blob_conf.shape(), var_axis, &instance_shape);
      const int64_t elem_cnt = instance_shape.elem_cnt();
      if (!blob->blob_desc().is_dynamic()) {
        const int64_t exp_elem_cnt =
            std::accumulate(blob_conf.shape().dim().begin(), blob_conf.shape().dim().end(), 1,
                            std::multiplies<int64_t>());
        CHECK_EQ(elem_cnt, exp_elem_cnt);
      }
      MultiThreadLoop(batch_data->size(), [&](int64_t n) {
        const DataField* data_field = batch_data->at(n).GetField(blob_conf.data_case());
        size_t elem_bytes_size = GetSizeOfDataType(blob_conf.data_type());
        data_field->ToBuffer(dptr + n * elem_cnt * elem_bytes_size, blob_conf.data_type());
        Shape shape;
        data_field->InferShape(blob_conf.shape(), var_axis, &shape);
        CHECK(instance_shape == shape);
      });
      DimVector shape_vec(instance_shape.dim_vec());
      shape_vec.insert(shape_vec.begin(), batch_data->size());
      auto* mut_shape_view = blob->mut_shape_view();
      if (mut_shape_view) { mut_shape_view->set_shape(Shape(shape_vec)); }
    } else {
      TensorBackInserter tensor_inserter(blob);
      tensor_inserter.ReserveOneEmptyTensorList();
      for (DataInstance& data_inst : *batch_data) {
        auto* tensor = tensor_inserter.add_tensor();
        CHECK_EQ(tensor->dptr(), dptr);
        const DataField* data_field = data_inst.GetField(blob_conf.data_case());
        size_t written_size = data_field->ToBuffer(dptr, blob_conf.data_type());
        dptr += written_size;

        Shape inst_shape;
        data_field->InferShape(blob_conf.shape(), var_axis, &inst_shape);
        DimVector dim_vec(inst_shape.dim_vec());
        dim_vec.insert(dim_vec.begin(), 1);
        tensor->set_shape(ShapeView(dim_vec.data(), dim_vec.size()));
        CHECK_EQ(tensor->ByteSize(), written_size);
      }
      CHECK_EQ(blob->total_num_of_tensors(), blob->static_shape().At(0));
    }
  }

  std::unique_ptr<detection::DataLoader> data_loader_;
};

REGISTER_KERNEL(OperatorConf::kDetectionDataLoadConf, DetectionDataLoadKernel);

}  // namespace oneflow
