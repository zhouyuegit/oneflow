import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())


def _check_unique(test_case, x, y, idx, count):
    ref_y, ref_count = np.unique(x, return_counts=True)
    sorted_idx = np.argsort(ref_y)
    ref_y = ref_y[sorted_idx]
    ref_count = ref_count[sorted_idx]
    test_case.assertTrue(y.shape, ref_y.shape)
    test_case.assertTrue(np.array_equal(y[idx], x))
    sorted_idx = np.argsort(y)
    test_case.assertTrue(np.array_equal(ref_y, y[sorted_idx]))
    test_case.assertTrue(np.array_equal(count[sorted_idx], ref_count))


def _run_test(test_case, x, dtype, device):
    @flow.function(func_config)
    def UniqueWithCountsJob(x=flow.FixedTensorDef(x.shape, dtype=dtype)):
        with flow.fixed_placement(device, "0:0"):
            return flow.unique_with_counts(x, resize_output=(device == 'gpu'))

    if device == 'gpu':
        y, idx, count = UniqueWithCountsJob(x).get()
        _check_unique(test_case, x, y.ndarray_list()[0], idx.ndarray(), count.ndarray_list()[0])
    else:
        y, idx, count, num_unique = UniqueWithCountsJob(x).get()
        y = y.ndarray()[0:num_unique.ndarray().item()]
        count = count.ndarray()[0:num_unique.ndarray().item()]
        _check_unique(test_case, x, y, idx.ndarray(), count)


def test_unique_with_counts_int(test_case):
    x = np.asarray(list(range(32)) * 2).astype(np.int32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'gpu')


def test_unique_with_counts_float(test_case):
    x = np.asarray(list(range(32)) * 2).astype(np.float32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.float32, 'gpu')


def test_unique_with_counts_random_gpu(test_case):
    x = np.random.randint(0, 32, 1024).astype(np.int32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'gpu')


def test_unique_with_counts_random_cpu(test_case):
    x = np.random.randint(0, 32, 1024).astype(np.int32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'cpu')
