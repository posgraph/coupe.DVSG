import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

def scale_RGB(rgb):
    RESNET_MEAN = [103.939, 116.779, 123.68]
    rgb_scaled = rgb * 255.0
    red, green, blue = tf.split(rgb_scaled, 3, 3)
    bgr = tf.concat([
        blue - RESNET_MEAN[0],
        green - RESNET_MEAN[1],
        red - RESNET_MEAN[2],
    ], axis=3)

    return bgr

def localizationNet(input, param_dim, is_train = False, reuse = False, scope = 'resnet_v1_50'):
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope(scope, reuse = reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, _ = resnet_v1.resnet_v1_50(scale_RGB(input), global_pool = True, is_training=is_train, reuse = reuse)

        net = tl.layers.InputLayer(net)
        net = tl.layers.FlattenLayer(net, name = 'flatten')
        net = tl.layers.DenseLayer(net, n_units = 2048, act = lrelu, name='df/dense1')
        net = tl.layers.DenseLayer(net, n_units = 1024, act = lrelu, name='df/dense2')
        net = tl.layers.DenseLayer(net, n_units = 512, act = lrelu, name='df/dense3')
        net = tl.layers.DenseLayer(net, n_units = 50, act = tf.identity, name='df/dense4')

        thetas_affine = net.outputs
        thetas_affine = tf.reshape(thetas_affine, [-1, param_dim, 2])

    return thetas_affine