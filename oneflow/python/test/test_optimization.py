import oneflow as flow
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_util

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)


def calc_per_img():
    flow.clear_default_session()

    @flow.function
    def CalcPerImgJob():
        boxes = flow.get_variable(
            name="boxes",
            shape=(120000, 4),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0.0, maxval=1.0, dtype=flow.float32),
        )
        labels = flow.get_variable(
            name="labels",
            shape=(120000,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=80, dtype=flow.int32),
        )
        scores = flow.get_variable(
            name="scores",
            shape=(120000,),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0.0, maxval=1.0, dtype=flow.float32),
        )
        matched_indices = flow.get_variable(
            name="matched_indices",
            shape=(120000,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=100, dtype=flow.int32),
        )
        box_indices = flow.get_variable(
            name="box_indices",
            shape=(256,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=11999, dtype=flow.int32),
        )
        pos_box_indices = flow.get_variable(
            name="pos_box_indices",
            shape=(64,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=11999, dtype=flow.int32),
        )
        for i in range(6):
          flow.local_gather(boxes, pos_box_indices)
        for i in range(2):
          flow.local_gather(scores, box_indices)
        for i in range(2):
          flow.local_gather(labels, box_indices)
        for i in range(2):
          flow.local_gather(matched_indices, box_indices)
        return None
    
    check_point = flow.train.CheckPoint()
    check_point.init()
    CalcPerImgJob()

def calc_per_batch():
    flow.clear_default_session()

    @flow.function
    def CalcPerBatchJob():
        boxes = flow.get_variable(
            name="boxes",
            shape=(240000, 4),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0.0, maxval=1.0, dtype=flow.float32),
        )
        labels = flow.get_variable(
            name="labels",
            shape=(240000,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=80, dtype=flow.int32),
        )
        scores = flow.get_variable(
            name="scores",
            shape=(240000,),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0.0, maxval=1.0, dtype=flow.float32),
        )
        matched_indices = flow.get_variable(
            name="matched_indices",
            shape=(240000,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=100, dtype=flow.int32),
        )
        box_indices = flow.get_variable(
            name="box_indices",
            shape=(512,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=11999, dtype=flow.int32),
        )
        pos_box_indices = flow.get_variable(
            name="pos_box_indices",
            shape=(128,),
            dtype=flow.int32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=11999, dtype=flow.int32),
        )
        for i in range(3):
          flow.local_gather(boxes, pos_box_indices)
        flow.local_gather(scores, box_indices)
        flow.local_gather(labels, box_indices)
        flow.local_gather(matched_indices, box_indices)
        return None
    
    check_point = flow.train.CheckPoint()
    check_point.init()
    CalcPerBatchJob()


if __name__ == "__main__":
    calc_per_img()
    calc_per_batch()
