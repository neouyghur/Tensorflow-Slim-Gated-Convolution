import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
slim = tf.contrib.slim

@add_arg_scope
def gated_conv2d(*args, **kwargs):
    """ gated convolution with special padding.

    Args:
        *args:
        **kwargs:

    Returns:

    """
    padding = kwargs['padding']
    rate = kwargs['rate']
    ksize = args[2][0]
    x = args[0]
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'
    kwargs['padding'] = padding
    args = list(args)
    args[0] = x
    output = slim.conv2d(*args, **kwargs)
    x1, x2 = tf.split(output, 2, axis=-1)  # split along channels
    return tf.nn.sigmoid(x2) * tf.nn.leaky_relu(x1)

@add_arg_scope
def gated_NNdeconv2d(*args, **kwargs):
    """ Gated nearest neighbor deconvolution (transposed convolution)
    """
    args=list(args)
    input_shape = args[0].get_shape().as_list()
    new_input_shape = [int(input_shape[1] * 2), int(input_shape[2] * 2)] # scale is 2
    args[0] = tf.image.resize_nearest_neighbor(args[0], new_input_shape, align_corners=True, name='resize')
    return gated_conv2d(*args, **kwargs)
