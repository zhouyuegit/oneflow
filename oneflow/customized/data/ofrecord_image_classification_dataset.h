/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_

#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/customized/data/ofrecord_dataset.h"
#include "oneflow/customized/image/image_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace data {

struct ImageClassificationDataInstance {
  std::shared_ptr<TensorBuffer> label;
  std::shared_ptr<TensorBuffer> image;
};

class OFRecordImageClassificationDataset final : public Dataset<ImageClassificationDataInstance> {
 public:
  using LoadTargetPtr = std::shared_ptr<ImageClassificationDataInstance>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OF_DISALLOW_COPY_AND_MOVE(OFRecordImageClassificationDataset);
  explicit OFRecordImageClassificationDataset(user_op::KernelInitContext* ctx) : shutdown_(false) {
    const std::string& color_space = ctx->Attr<std::string>("color_space");
    const std::string& image_feature_key = ctx->Attr<std::string>("image_feature_key");
    const std::string& label_feature_key = ctx->Attr<std::string>("label_feature_key");
    const int64_t buffer_size = ctx->Attr<int64_t>("buffer_size");
    underlying_.reset(new OFRecordDataset(ctx));
    instance_buffer_.reset(new Buffer<LoadTargetPtr>(buffer_size));
    decode_pool_.reset(
        new ThreadPool(Global<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize()));
    load_thread_ = std::thread([&] {
      while (!shutdown_) {
        OFRecordDataset::LoadTargetPtrList buffers = underlying_->Next();
        for (const auto& buffer : buffers) {
          decode_pool_->AddWork([&] {
            OFRecord record;
            record.ParseFromArray(buffer->data<char>(), buffer->shape().elem_cnt());
            std::shared_ptr<ImageClassificationDataInstance> instance;
            instance->image.reset(new TensorBuffer());
            auto image_feature_it = record.feature().find(image_feature_key);
            CHECK(image_feature_it != record.feature().end());
            const Feature& image_feature = image_feature_it->second;
            CHECK(image_feature.has_bytes_list());
            CHECK(image_feature.bytes_list().value_size() == 1);
            const std::string& src_data = image_feature.bytes_list().value(0);
            cv::Mat image = cv::imdecode(
                cv::Mat(1, src_data.size(), CV_8UC1, (void*)(src_data.data())), cv::IMREAD_COLOR);
            int W = image.cols;
            int H = image.rows;

            // convert color space
            if (ImageUtil::IsColor(color_space) && color_space != "BGR") {
              ImageUtil::ConvertColor("BGR", image, color_space, image);
            }

            CHECK(image.isContinuous());
            const int c = ImageUtil::IsColor(color_space) ? 3 : 1;
            CHECK_EQ(c, image.channels());
            Shape image_shape({H, W, c});
            instance->image->Resize(image_shape, DataType::kUInt8);
            CHECK_EQ(image_shape.elem_cnt(), instance->image->nbytes());
            CHECK_EQ(image_shape.elem_cnt(), image.total() * image.elemSize());
            memcpy(instance->image->mut_data<uint8_t>(), image.ptr(), image_shape.elem_cnt());

            auto label_feature_it = record.feature().find(label_feature_key);
            CHECK(label_feature_it != record.feature().end());
            const Feature& label_feature = label_feature_it->second;
            instance->label.reset(new TensorBuffer());
            instance->label->Resize(Shape({1}), DataType::kInt32);
            if (label_feature.has_int32_list()) {
              CHECK_EQ(label_feature.int32_list().value_size(), 1);
              *instance->label->mut_data<int32_t>() = label_feature.int32_list().value(0);
            } else if (label_feature.has_int64_list()) {
              CHECK_EQ(label_feature.int64_list().value_size(), 1);
              *instance->label->mut_data<int32_t>() = label_feature.int64_list().value(0);
            } else {
              UNIMPLEMENTED();
            }
            instance_buffer_->Send(instance);
          });
        }
      }
    });
  }
  ~OFRecordImageClassificationDataset() override {
    shutdown_ = true;
    instance_buffer_->Close();
    load_thread_.join();
    decode_pool_.reset();
  }

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    LoadTargetPtr sample_ptr;
    instance_buffer_->Receive(&sample_ptr);
    ret.push_back(std::move(sample_ptr));
    return ret;
  }

 private:
  std::unique_ptr<OFRecordDataset> underlying_;
  std::unique_ptr<ThreadPool> decode_pool_;
  std::unique_ptr<Buffer<LoadTargetPtr>> instance_buffer_;
  std::thread load_thread_;

  std::atomic<bool> shutdown_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_
