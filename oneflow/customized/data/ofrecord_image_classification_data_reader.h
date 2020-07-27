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
#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/ofrecord_dataset.h"
#include "oneflow/customized/data/ofrecord_parser.h"
#include "oneflow/customized/data/random_shuffle_dataset.h"
#include "oneflow/customized/data/batch_dataset.h"
#include "oneflow/customized/data/ofrecord_image_classification_dataset.h"
#include "oneflow/customized/data/ofrecord_image_classification_parser.h"
#include <iostream>

namespace oneflow {

namespace data {

class OFRecordImageClassificationDataReader final
    : public DataReader<ImageClassificationDataInstance> {
 public:
  OFRecordImageClassificationDataReader(user_op::KernelInitContext* ctx)
      : DataReader<ImageClassificationDataInstance>(ctx) {
    loader_.reset(new OFRecordImageClassificationDataset(ctx));
    parser_.reset(new OFRecordImageClassificationParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(
          new RandomShuffleDataset<ImageClassificationDataInstance>(ctx, std::move(loader_)));
    }
    const int64_t batch_size = ctx->TensorDesc4ArgNameAndIndex("image", 0)->shape().elem_cnt();
    loader_.reset(
        new BatchDataset<ImageClassificationDataInstance>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~OFRecordImageClassificationDataReader() override = default;

 protected:
  using DataReader<ImageClassificationDataInstance>::loader_;
  using DataReader<ImageClassificationDataInstance>::parser_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATA_READER_H_
