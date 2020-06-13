import oneflow as flow


def deconv2d(input, filters, size, name, strides=2,
             trainable=True, reuse=False, const_init=False, use_bias=False):
    name_ = name if reuse == False else name + "_reuse"
    # weight : [in_channels, out_channels, height, width]
    weight_shape = (input.shape[1], filters, size, size)
    output_shape = (input.shape[0],
                    input.shape[1],
                    input.shape[2] * strides,
                    input.shape[3] * strides)

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02) \
            if not const_init else flow.constant_initializer(0.002),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose(input, weight, strides=[strides, strides], 
                                      output_shape=output_shape, padding="SAME",
                                      data_format="NCHW", name=name_)
    
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def conv2d(input, filters, size, name, strides=2, padding='same',
           trainable=True, reuse=False, const_init=False, use_bias=True):
    name_ = name if reuse == False else name + "_reuse"

    # (output_dim, k_h, k_w, input.shape[3]) if NHWC
    weight_shape = (filters, input.shape[1], size, size) 
    weight = flow.get_variable(name + "-weight",
                            shape=weight_shape,
                            dtype=input.dtype,
                            initializer=flow.random_normal_initializer(stddev=0.02) \
                                if not const_init else flow.constant_initializer(0.002),
                            trainable=trainable,
                            )

    output = flow.nn.compat_conv2d(input, weight, strides=[strides, strides], 
                            padding=padding, data_format="NCHW", name=name_)

    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output

def batchnorm(input, name, reuse=False):
    if reuse: name = name + "_reuse"
    return flow.layers.batch_normalization(input, axis=1, name=name)
