import oneflow as flow
import numpy as np
import random


def _distribute_split_tensor_buffer(test_case, shape=(2, 10, 4)):
    assert isinstance(shape, (list, tuple))
    assert len(shape) >= 2

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    # func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ds_tb_fn(input_def=flow.MirroredTensorListDef(shape=shape, dtype=flow.float, batch_axis=0)):
        tb = flow.tensor_list_to_tensor_buffer(input_def)
        with flow.distribute.mirrored_strategy():
            output = flow.identity(tb)
            return flow.tensor_buffer_to_tensor_list(output, shape=shape[1:], dtype=flow.float)

    rand_input_shape_list = [
        [1] + [random.randrange(1, shape[1])] + list(shape[2:]) for _ in range(shape[0])
    ]
    rand_input_list = [
        np.random.randn(*input_shape).astype(np.single) for input_shape in rand_input_shape_list
    ]
    output = ds_tb_fn([rand_input_list]).get().ndarray_lists()[0]

    print("input:\n", rand_input_list)
    print("output:\n", output)
    for i, o in zip(rand_input_list, output):
        test_case.assertTrue(np.array_equal(i, o))


def test_distribute_split_tensor_buffer(test_case):
    _distribute_split_tensor_buffer(test_case)
