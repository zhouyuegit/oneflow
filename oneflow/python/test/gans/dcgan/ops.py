import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import os
import numpy as np
from PIL import Image

def get_const_initializer():
    return flow.constant_initializer(0.002)

bag = {}
def deconv2d(input, output_shape,
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
             name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    name_ = name if reuse == False else name + "_reuse"
    # weight : [in_channels, out_channels, height, width]
    weight_shape = (input.static_shape[1], output_shape[1], k_h, k_w)

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose_V2(input, weight, strides=[d_h, d_w], output_shape=output_shape,
                                         padding="SAME", data_format="NCHW", name=name_)
    
    bias = flow.get_variable(
        name + "-bias",
        shape=(output_shape[1],),
        dtype=input.dtype,
        initializer=flow.constant_initializer(0.0),
        trainable=trainable,
    )
    output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def conv2d(input, output_dim, 
       k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
       name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    name_ = name if reuse == False else name + "_reuse"
    # name_ = id_util.UniqueStr(name + "_")

    weight_shape = (output_dim, input.static_shape[1], k_h, k_w) # (output_dim, k_h, k_w, input.static_shape[3]) if NHWC
    if reuse:
        assert name + "-weight" in bag
        weight = bag[name + "-weight"]
    else:
        weight = flow.get_variable(name_ + "-weight",
                                shape=weight_shape,
                                dtype=input.dtype,
                                initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
                                trainable=trainable,
                                )
        bag.update({name + "-weight": weight})

    output = flow.nn.conv2d(input, weight, strides=[d_h, d_w], 
                            padding="SAME", data_format="NCHW", name=name_)

    if reuse:
        assert name + "-bias" in bag
        bias = bag[name + "-bias"]
    else:
        bias = flow.get_variable(
            name_ + "-bias",
            shape=(output_dim,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
        )
        bag.update({name + "-bias": bias})

    output = flow.nn.bias_add(output, bias, "NCHW")
    return output

# def batch_norm(input, name=None, trainable=True, reuse=False, const_init=False):
#     assert name is not None
#     name_ = name if reuse == False else name + "_reuse"
#     return flow.layers.batch_normalization(
#         inputs=input,
#         axis=1,
#         momentum=0.997,
#         epsilon=0.00002,
#         center=True,
#         scale=True,
#         trainable=trainable,
#         name=name_,
#     )

def batch_norm(input, name=None, trainable=True, reuse=False, const_init=False):
    # do nothing
    return input


def linear(input, units, name=None, trainable=True, reuse=False, const_init=False):
    assert name is not None
    # name_ = id_util.UniqueStr(name + "_")
    name_ = name if reuse == False else name + "_reuse"

    in_shape = input.static_shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    inputs = (
        flow.reshape(input, (-1, in_shape[-1])) if in_num_axes > 2 else input
    )

    if reuse:
        assert name + "-weight" in bag
        weight = bag[name + "-weight"]
    else:
        weight = flow.get_variable(
            name="{}-weight".format(name_),
            shape=(units, inputs.static_shape[1]),
            dtype=inputs.dtype,
            initializer=flow.random_normal_initializer(stddev=0.02) if not const_init else get_const_initializer(),
            trainable=trainable,
            model_name="weight",
        )
        bag.update({name + "-weight": weight})

    out = flow.matmul(
        a=inputs,
        b=weight,
        transpose_b=True,
        name=name_ + "matmul",
    )

    if reuse:
        assert name + "-bias" in bag
        bias = bag[name + "-bias"]
    else:
        bias = flow.get_variable(
            name="{}-bias".format(name_),
            shape=(units,),
            dtype=inputs.dtype,
            initializer=flow.random_normal_initializer() if not const_init else get_const_initializer(),
            trainable=trainable,
            model_name="bias",
        )
        bag.update({name + "-bias": bias})

    out = flow.nn.bias_add(
        out, bias, name=name_ + "_bias_add"
    )

    out = (
        flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
    )
    return out

def relu(input):
    return flow.keras.activations.relu(input)

def tanh(input):
    return flow.keras.activations.tanh(input)

def lrelu(input, alpha=0.2):
    name = id_util.UniqueStr("Leaky_Relu_")
    return flow.user_op_builder(name).Op("leaky_relu").Input("x",[input]).Output("y") \
           .SetAttr("alpha", alpha, "AttrTypeFloat").Build().RemoteBlobList()[0]

def load_mnist(data_dir='./data', dataset_name='mnist', transpose=True):
    data_dir = os.path.join(data_dir, dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    if transpose:
        X = np.transpose(X, (0,3,1,2))
        X = np.pad(X, ((0,),(0,),(2,),(2,)), "edge")

    return X/255., y_vec
