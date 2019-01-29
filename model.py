import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

import collections

from warp_with_optical_flow import *
from spatial_transformer import *
from networks import *

class StabNet:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.c = 3
        self.control_point_num = 5 # grid_num (4) + 1
        self.reuse = collections.OrderedDict()
        self.elastic_transformer = ElasticTransformer(out_size = [self.h, self.w], param_dim = 2 * self.control_point_num ** 2, param_dim_per_side = self.control_point_num)


    def init_pretrain_inputs(self, sample_num):
        self.sample_num = sample_num
        self.inputs_pretrain = self.init_train_inputs(self.sample_num)
        return self.inputs_pretrain

    def get_pretrain_model(self, is_train):
        self.inputs = self.inputs_pretrain
        return self.get_train_model(is_train)

    def init_train_inputs(self, sample_num):
        self.sample_num = sample_num
        with tf.variable_scope('input'):
            self.inputs = collections.OrderedDict()
            self.inputs['patches_t_1'] = tf.placeholder('float32', [None, None, None, 3 * self.sample_num], name = 'input_frames_t_1')
            self.inputs['patches_t'] = tf.placeholder('float32', [None, None, None, 3 * self.sample_num], name = 'input_frames_t')

            self.inputs['s_t_1_gt'] = tf.placeholder('float32', [None, None, None, 3], name = 'stable_frame_t_1_gt')
            self.inputs['s_t_gt'] = tf.placeholder('float32', [None, None, None, 3], name = 'stable_frame_t_gt')

            self.inputs['u_t_1'] = tf.placeholder('float32', [None, None, None, 3], name = 'unstable_frame_t_1')
            self.inputs['u_t'] = tf.placeholder('float32', [None, None, None, 3], name = 'unstable_frame_t')

            self.inputs['of_t'] = tf.placeholder('float32', [None, None, None, 2], name = 'optical_flow_t')

            self.inputs['surfs_t_1'] = tf.placeholder('int32', [None, 2, None, 2], name = 'surfs_t_1')
            self.inputs['surfs_t'] = tf.placeholder('int32', [None, 2, None, 2], name = 'surfs_t')

            self.inputs['surfs_dim_t_1'] = tf.placeholder('float32', [None], name = 'surfs_dim_t_1')
            self.inputs['surfs_dim_t'] = tf.placeholder('float32', [None], name = 'surfs_dim_t')

        return self.inputs

    def get_train_model(self, is_train):
        outputs = collections.OrderedDict()
        outputs['patches_masked_t_1'], outputs['random_masks_t_1'] = self.random_mask(self.inputs['patches_t_1'], [self.h, self.w], self.sample_num)
        outputs['patches_masked_t'], outputs['random_masks_t'] = self.random_mask(self.inputs['patches_t'], [self.h, self.w], self.sample_num)

        with tf.variable_scope('stabNet') as scope:
            ## Regressor
            outputs['F_t_1'] = localizationNet(outputs['patches_masked_t_1'], is_train, self.get_reuse('stabNet'), scope = scope)
            outputs['F_t'] = localizationNet(outputs['patches_masked_t'], is_train, self.get_reuse('stabNet'), scope = scope)

            ## STN
            stl_affine = ProjectiveTransformer([self.h, self.w])

            outputs['s_t_1_pred'], outputs['x_offset_t_1'], outputs['y_offset_t_1'] = self.elastic_transformer.transform(self.inputs['u_t_1'], outputs['F_t_1'])
            outputs['s_t_1_pred_mask'], _, _ = self.elastic_transformer.transform(tf.ones_like(self.inputs['u_t_1']), outputs['F_t_1'])

            outputs['s_t_pred'], outputs['x_offset_t'], outputs['y_offset_t'] = self.elastic_transformer.transform(self.inputs['u_t'], outputs['F_t'])
            outputs['s_t_pred_mask'], _, _ = self.elastic_transformer.transform(tf.ones_like(self.inputs['u_t']), outputs['F_t'])
            outputs['s_t_pred_warped'] = tf_warp(outputs['s_t_pred'], self.inputs['of_t'], self.h, self.w)
            outputs['s_t_pred_warped_mask'] = tf_warp(outputs['s_t_pred_mask'], self.inputs['of_t'], self.h, self.w)

        return outputs

    def get_evaluation_model(self, sample_num):
        self.sample_num = sample_num
        with tf.variable_scope('input'):
            ### PLACE HOlDERS ###
            inputs = collections.OrderedDict()
            inputs['patches'] = tf.placeholder('float32', [None, None, None, 3 * 2 * self.sample_num], name = 'input_frames_t')
            inputs['u'] = tf.placeholder('float32', [None, None, None, 3], name = 'unstable_frame_t')

            batch_size = tf.shape(inputs['u'])[0]
            h = tf.shape(inputs['u'])[1]
            w = tf.shape(inputs['u'])[2]

            patches_masked, _ = self.random_mask(inputs['patches'], [h, w], self.sample_num, True)

        with tf.variable_scope('main_net') as scope:
            with tf.variable_scope('stabNet') as scope:
                F = self.localizationNet(patches_masked, is_train = False, reuse = False, scope = scope)

            with tf.variable_scope('spatial_transformer') as scope:
                stl_affine = ProjectiveTransformer([h, w])
                s_pred = stl_affine.transform(inputs['u'], F)

        self.inputs = inputs
        self.output = s_pred
        return self.inputs, self.output

    def init_vars(self, sess):
        exclude_scope = 'stabNet/resnet_v1_50/conv1'
        variables_to_restore = collections.OrderedDict()
        for var in slim.get_model_variables():
            if var.op.name.startswith(exclude_scope) == False:
                variables_to_restore[var.op.name.replace('stabNet/', '')] = var
        init_function = slim.assign_from_checkpoint_fn('./pretrained/resnet_v1_50.ckpt', variables_to_restore, ignore_missing_vars=True)
        init_function(sess)

    def random_mask(self, patches, out_size, sample_num):
        mask_affine = ProjectiveTransformer(out_size)
        batch_size = tf.shape(patches)[0]
        mask = tf.ones_like(patches)
        H = tf.random_uniform([batch_size, 8], minval = -1, maxval = 1)
        H = H * tf.constant([0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1])
        H = H + tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mask = mask_affine.transform(mask, H)

        return patches * mask, mask

    def get_reuse(self, scope):
        if scope in list(self.reuse.keys()):
            self.reuse[scope] = True if self.reuse[scope] is False else True
        else:
            self.reuse[scope] = False

        return self.reuse[scope]

    def get_vars_train(self):
        return self._get_vars('stabNet', True, False)

    def get_save_vars_train(self):
        return self._get_vars('stabNet', False, False)

    def get_vars_pretrain(self):
        return self._get_vars('stabNet', True, False)

    def get_save_vars_pretrain(self):
        return self._get_vars('stabNet', False, False)

    def _get_vars(self, name, train_only, verbose, exclude = None):
        if exclude is None:
            return tl.layers.get_variables_with_name(name, train_only, verbose)
        else:
            return [var for var in tl.layers.get_variables_with_name(name, train_only, verbose) if exclude not in var.name]
