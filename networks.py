import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2

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

def featureNorm(f):
    eps = 1e-6
    f = f - tf.reduce_min(f, axis = 3, keep_dims = True)
    f = tf.div(f, (tf.reduce_max(f, axis = 3, keep_dims = True) + eps))
    return f

def featureL2Norm(f):
    eps = 1e-6
    return tf.div(f, tf.expand_dims(tf.pow(tf.reduce_sum(tf.pow(f, 2), 3) + eps, 0.5), axis = 3))

######################################################################################################################

def localizationNet(input, param_dim, is_train = False, reuse = False, scope = 'localizationNet'):
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

######################################################################################################################

def correlationNet(frames_src, frames_trg, feature_norm, model = '101', reuse = False, scope = 'correlationNet', is_pretrain = False):
    with tf.variable_scope(scope, reuse = reuse) as scope:
        feats_src = featureExtractor(frames_src, feature_norm, model, reuse = reuse, scope = 'featureExtractor')
        feats_trg = featureExtractor(frames_trg, feature_norm, model, reuse = True, scope = 'featureExtractor')

        CMs = correlationLayer(feats_src, feats_trg, feature_norm)

    return CMs

def featureExtractor(input, feature_norm, model = '101', reuse = False, scope = 'resnet_v2_101'):
    with tf.variable_scope(scope, reuse = reuse) as scope:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            input = scale_RGB(input)
            if '50' in model:
                net, end_points  = resnet_v2.resnet_v2_50(input, global_pool = True, is_training = False, reuse = reuse)
            elif '101' in model:
                _, end_points  = resnet_v2.resnet_v2_101(input, global_pool = True, is_training = False, reuse = reuse)
                #f = end_points['stabNet/pathFinder/featureExtractor/resnet_v2_101/block3/unit_23/bottleneck_v2/conv1'] #(18 X 32) X (18 X 32)
                f = end_points['{}/resnet_v2_101/block4/unit_2/bottleneck_v2/conv1'.format(scope.name)] #(9 X 16) X (9 X 16)
        if feature_norm:
            f = featureL2Norm(f)

    return f

def correlationLayer(f_src, f_trg, feature_norm, f_src_mask = None, f_trg_mask = None):
        b = tf.shape(f_src)[0]
        h = tf.shape(f_src)[1]
        w = tf.shape(f_src)[2]
        c = tf.shape(f_src)[3]

        #b, h, w, c = f_src.get_shape().as_list()[:]

        f_src = tf.reshape(tf.transpose(f_src, [0, 3, 2, 1]), [-1, c, w * h])
        f_trg = tf.reshape(f_trg, [-1, h * w, c])
        correlation = tf.matmul(f_trg, f_src)
        correlation = tf.reshape(correlation, [-1, h, w, w * h], name = 'correlation') # [-1, trg, src, cor]
        if feature_norm:
            correlation = featureL2Norm(tf.nn.relu(correlation))
            #correlation = tf.nn.relu(correlation)

        correlation = tf.reshape(correlation, [-1, h * w, w * h, 1], name = 'correlation') # [-1, trg, src, cor]
        if feature_norm:
            correlation = correlation - tf.reduce_min(correlation, axis = [1, 2, 3], keepdims = True)
            correlation = correlation / tf.reduce_max(correlation, axis = [1, 2, 3], keepdims = True)

        return correlation            
