# Tensorflow-Slim-Gated-Convolution


```
from gated_convolution import *

with slim.arg_scope([slim.batch_norm], is_training=phase_train):
    with slim.arg_scope([gated_conv2d, gated_NNdeconv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=config.G_STDDEV),
                        weights_regularizer=slim.l2_regularizer(config.G_REGULARIZE_LAMDA),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([gated_conv2d], padding='SYMMETRIC', rate=1, activation_fn=None):
            # slim.conv2d default relu activation
            # subsampling
            conv0 = gated_conv2d(image, 64, [5,5], 2, scope='conv0', normalizer_fn=None)
            
        with slim.arg_scope([gated_conv2d, gated_NNdeconv2d],
                            stride=1, rate=1, padding='SYMMETRIC', activation_fn=None):
            # upsampling
            conv_t0 = gated_NNdeconv2d(conv0, 64, [3,3], scope='conv_t0')

```
