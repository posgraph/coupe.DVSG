import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np

import collections

from warp_with_optical_flow import *
from ThinPlateSpline import ThinPlateSpline as stn
from spatial_transformer import ProjectiveTransformer
from networks import *

class StabNet:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.c = 3
        self.num_control_points = 5
        self.param_dim = self.num_control_points ** 2
        self.feature_norm = True

        self.stabNet_model = 'resnet_v1_50'
        self.correlationNet_model = 'resnet_v2_101' #resnet_v2_50

        self.reuse = collections.OrderedDict()

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

        outputs['V_src'] = np.array([ # source position
          [-1, -1],[-0.5, -1],[0, -1],[0.5, -1],[1, -1],
          [-1, -0.5],[-0.5, -0.5],[0, -0.5],[0.5, -0.5],[1, -0.5],
          [-1, 0],[-0.5, 0],[0, 0],[0.5, 0],[1, 0],
          [-1, 0.5],[-0.5, 0.5],[0, 0.5],[0.5, 0.5],[1, 0.5],
          [-1, 1],[-0.5, 1],[0, 1],[0.5, 1],[1, 1]])
        outputs['V_src'] = tf.tile(tf.constant(outputs['V_src'].reshape([1, self.param_dim, 2]), dtype=tf.float32), [tf.shape(self.inputs['u_t_1'])[0], 1, 1])
        outputs['num_control_points'] = self.num_control_points

        outputs['patches_masked_t_1'], outputs['random_masks_t_1'] = self.random_mask(self.inputs['patches_t_1'], [self.h, self.w], self.sample_num)
        outputs['patches_masked_t'], outputs['random_masks_t'] = self.random_mask(self.inputs['patches_t'], [self.h, self.w], self.sample_num)

        with tf.variable_scope('stabNet'):
            ## Regressor
            with tf.variable_scope('localizationNet') as scope:
                outputs['F_t_1'] = localizationNet(outputs['patches_masked_t_1'], self.param_dim, is_train, self.get_reuse('localizationNet'), scope = scope)
                outputs['F_t'] = localizationNet(outputs['patches_masked_t'], self.param_dim, is_train, self.get_reuse('localizationNet'), scope = scope)

            ## STN
            outputs['s_t_1_pred'], outputs['x_offset_t_1'], outputs['y_offset_t_1'] = stn(self.inputs['u_t_1'], outputs['V_src'], outputs['F_t_1'], [self.h, self.w])
            outputs['s_t_1_pred_mask'], _, _ = stn(tf.ones_like(self.inputs['u_t_1']), outputs['V_src'], outputs['F_t_1'], [self.h, self.w])

            outputs['s_t_pred'], outputs['x_offset_t'], outputs['y_offset_t'] = stn(self.inputs['u_t'], outputs['V_src'], outputs['F_t'], [self.h, self.w])
            outputs['s_t_pred_mask'], _, _ = stn(tf.ones_like(self.inputs['u_t']), outputs['V_src'], outputs['F_t'], [self.h, self.w])

            outputs['s_t_1_gt_warp'] = tf_warp(self.inputs['s_t_1_gt'], self.inputs['of_t'], self.h, self.w)
            outputs['s_t_gt_warp'] = tf_warp(self.inputs['s_t_gt'], self.inputs['of_t'], self.h, self.w)

            with tf.variable_scope('correlationNet') as scope:
                outputs['CM_t_1_pred'] = correlationNet(outputs['s_t_1_pred'], self.inputs['s_t_1_gt'], self.feature_norm, self.correlationNet_model, reuse = self.get_reuse('correlationNet'), scope = scope)
                outputs['CM_t_1_gt'] = correlationNet(self.inputs['s_t_1_gt'] * outputs['s_t_1_pred_mask'], self.inputs['s_t_1_gt'], self.feature_norm, self.correlationNet_model, reuse = self.get_reuse('correlationNet'), scope = scope)
                outputs['CM_t_pred'] = correlationNet(outputs['s_t_pred'], self.inputs['s_t_gt'], self.feature_norm, self.correlationNet_model, reuse = self.get_reuse('correlationNet'), scope = scope)
                outputs['CM_t_gt'] = correlationNet(self.inputs['s_t_gt'] * outputs['s_t_pred_mask'], self.inputs['s_t_gt'], self.feature_norm, self.correlationNet_model, reuse = self.get_reuse('correlationNet'), scope = scope)

        return outputs

    def get_evaluation_model(self, sample_num):
        is_train = False
        inputs = collections.OrderedDict()
        inputs['patches_t'] = tf.placeholder('float32', [None, None, None, 3 * sample_num], name = 'input_frames_t')
        inputs['u_t'] = tf.placeholder('float32', [None, None, None, 3], name = 'unstable_frame_t')

        outputs = collections.OrderedDict()
        outputs['V_src'] = np.array([ # source position
          [-1, -1],[-0.5, -1],[0, -1],[0.5, -1],[1, -1],
          [-1, -0.5],[-0.5, -0.5],[0, -0.5],[0.5, -0.5],[1, -0.5],
          [-1, 0],[-0.5, 0],[0, 0],[0.5, 0],[1, 0],
          [-1, 0.5],[-0.5, 0.5],[0, 0.5],[0.5, 0.5],[1, 0.5],
          [-1, 1],[-0.5, 1],[0, 1],[0.5, 1],[1, 1]])
        outputs['V_src'] = tf.tile(tf.constant(outputs['V_src'].reshape([1, self.param_dim, 2]), dtype=tf.float32), [tf.shape(inputs['u_t'])[0], 1, 1])
        outputs['num_control_points'] = self.num_control_points

        with tf.variable_scope('stabNet'):
            ## Regressor
            with tf.variable_scope('localizationNet') as scope:
                outputs['F_t'] = localizationNet(inputs['patches_t'], self.param_dim, is_train, self.get_reuse('localizationNet'), scope = scope)

            ## STN
            outputs['s_t_pred'], outputs['x_offset_t'], outputs['y_offset_t'] = stn(inputs['u_t'], outputs['V_src'], outputs['F_t'], [self.h, self.w])
            outputs['s_t_pred_mask'], _, _ = stn(tf.ones_like(inputs['u_t']), outputs['V_src'], outputs['F_t'], [self.h, self.w])

        return inputs, outputs

    def init_vars(self, sess):
        exclude_scope = ['stabNet/localizationNet/resnet_v1_50/conv1']

        variables_to_restore_localizationNet = collections.OrderedDict()
        variables_to_restore_correlationNet = collections.OrderedDict()

        is_exclude = False

        for var in slim.get_model_variables():
            for exclude in exclude_scope:
                if var.op.name.startswith(exclude):
                    is_exclude = True

            if is_exclude:
               is_exclude = False
               continue

            if 'localizationNet' in var.op.name:
                variables_to_restore_localizationNet[var.op.name.replace('stabNet/localizationNet/', '')] = var
            elif 'correlationNet' in var.op.name:
                variables_to_restore_correlationNet[var.op.name.replace('stabNet/correlationNet/featureExtractor/', '')] = var

        if len(variables_to_restore_localizationNet) != 0:
            print('Initializing {}'.format(self.stabNet_model))
            init_function = slim.assign_from_checkpoint_fn('./pretrained/{}.ckpt'.format(self.stabNet_model), variables_to_restore_localizationNet, ignore_missing_vars=False)
            init_function(sess)
        if len(variables_to_restore_correlationNet) != 0:
            print('Initializing {}'.format(self.correlationNet_model))
            init_function = slim.assign_from_checkpoint_fn('./pretrained/{}.ckpt'.format(self.correlationNet_model), variables_to_restore_correlationNet, ignore_missing_vars=False)
            init_function(sess)

    def random_mask(self, patches, out_size, sample_num):
        mask_affine = ProjectiveTransformer(out_size)
        batch_size = tf.shape(patches)[0]

        mask = tf.ones_like(patches[:, :, :, : 3 * (sample_num - 1)])
        H = tf.random_uniform([batch_size, 8], minval = -1, maxval = 1)
        H = H * tf.constant([0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1])
        H = H + tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mask = mask_affine.transform(mask, H)
        mask = tf.concat([mask, tf.ones_like(patches[:, :, :, :3])], axis = 3)

        return patches * mask, mask

    def get_reuse(self, scope):
        if scope in list(self.reuse.keys()):
            self.reuse[scope] = True if self.reuse[scope] is False else True
        else:
            self.reuse[scope] = False

        return self.reuse[scope]

    def get_vars_train(self):
        return self._get_vars('localizationNet', True, False)

    def get_save_vars_train(self):
        return self._get_vars('localizationNet', False, False)

    def get_vars_pretrain(self):
        return self._get_vars('localizationNet', True, False)

    def get_save_vars_pretrain(self):
        return self._get_vars('localizationNet', False, False)

    def _get_vars(self, name, train_only, verbose, exclude = None):
        if exclude is None:
            return tl.layers.get_variables_with_name(name, train_only, verbose)
        else:
            return [var for var in tl.layers.get_variables_with_name(name, train_only, verbose) if exclude not in var.name]
