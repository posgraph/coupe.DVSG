import tensorflow as tf
import tensorlayer as tl

import collections
import numpy as np

from utils import *
from warp_with_optical_flow import *

class Trainer:
    def __init__(self, network, data_loaders, config):
        print(toGreen('Initializing network...'))
        self.network = network
        self.h = config.height
        self.w = config.width

        if config.is_pretrain:
            print(toRed('[PRETRAIN MODEL]'))
            self.sample_num = config.PRETRAIN.sample_num
            self.init_loss_coefficient(config.PRETRAIN)
            print(toGreen('Initializing model...'))
            self.inputs_pretrain = self.network.init_pretrain_inputs(self.sample_num)
            self.outputs_pretrain = self.network.get_pretrain_model(is_train = True)
            self.outputs_pretrain_test = self.network.get_pretrain_model(is_train = False)

            print(toGreen('Getting variables...'))
            self.pretraining_vars = self.network.get_vars_pretrain()
            self.pretraining_save_vars = self.network.get_save_vars_pretrain()

            print(toGreen('Initializing losses...'))
            self.build_loss_pretrain(self.outputs_pretrain, self.outputs_pretrain_test)

            print(toGreen('Initializing optim...'))
            self.build_pretrain_optim(config)
            self.build_pretrain_summary()

            print(toGreen('Registering dataloader to the network ...'))
            for data_loader in data_loaders:
                data_loader.init_data_loader(self.network, is_pretrain = True)

            print(toRed('pretraining variables ...'))
            for var in self.pretraining_vars:
                print(var.name)
            if config.pretrain_only:
                return

        print(toRed('[TRAIN MODEL]'))
        self.sample_num = config.sample_num
        self.init_loss_coefficient(config)
        print(toGreen('Initializing model...'))
        self.inputs = self.network.init_train_inputs(self.sample_num)
        self.outputs_train = self.network.get_train_model(is_train = True)
        self.outputs_test = self.network.get_train_model(is_train = False)

        print(toGreen('Getting variables...'))
        self.training_vars = self.network.get_vars_train()
        self.save_vars = self.network.get_save_vars_train()
        print(toGreen('Initializing losses...'))
        self.build_loss_train(self.outputs_train, self.outputs_test)

        print(toGreen('Initializing optim...'))
        self.build_train_optim(config)

        print(toGreen('Initializing summary...'))
        self.build_train_summary()

        print(toGreen('Registering dataloader to the network ...'))
        for data_loader in data_loaders:
            data_loader.init_data_loader(self.network, is_pretrain = False)

        print(toRed('training variables ...'))
        for var in self.training_vars:
            print(var.name)

    def build_loss_pretrain(self, outputs_train, outputs_test):
        self.pretrain_loss = collections.OrderedDict()
        with tf.name_scope('pretrain_loss'):
            self.pretrain_loss['identity'] = tl.cost.mean_squared_error(outputs_train['s_t_1_pred'], self.inputs_pretrain['u_t_1'], is_mean = True, name = 'loss_identity_t_1')\
                                            + tl.cost.mean_squared_error(outputs_train['s_t_pred'], self.inputs_pretrain['u_t'], is_mean = True, name = 'loss_identity_t')
            self.pretrain_loss = self.gather_only_applied_loss(self.pretrain_loss, self.loss_applied)
            self.pretrain_loss['total'] = tf.add_n([self.coef_placeholders[key] * val for (key, val) in self.pretrain_loss.items()], name = 'loss_total')
            print(toRed('{}'.format('applied losses: {}'.format([key for (key, val) in self.pretrain_loss.items()]))))

        self.pretrain_loss_test = collections.OrderedDict()
        with tf.name_scope('pretrain_loss'):
            self.pretrain_loss_test['identity'] = tl.cost.mean_squared_error(outputs_test['s_t_1_pred'], self.inputs_pretrain['u_t_1'], is_mean = True, name = 'loss_identity_t_1')\
                                            + tl.cost.mean_squared_error(outputs_test['s_t_pred'], self.inputs_pretrain['u_t'], is_mean = True, name = 'loss_identity_t')
            self.pretrain_loss_test = self.gather_only_applied_loss(self.pretrain_loss_test, self.loss_applied)
            self.pretrain_loss_test['total'] = tf.add_n([self.coef_placeholders[key] * val for (key, val) in self.pretrain_loss_test.items()], name = 'loss_total')

    def build_loss_train(self, outputs_train, outputs_test):
        batch_size = tf.shape(outputs_train['F_t'])[0]
        self.loss = collections.OrderedDict()
        with tf.name_scope('loss'):
            #self.loss['identity'] = tl.cost.mean_squared_error(outputs_train['F_t'], identity, is_mean = True, name = 'loss_identity_t')
            self.loss['stable'] = self.masked_MSE(outputs_train['s_t_pred'], self.inputs['s_t_gt'], outputs_train['s_t_pred_mask'], 'loss_stable')
            self.loss['temporal'] = self.temporal_loss(outputs_train['s_t_pred_warped'], outputs_train['s_t_1_pred'], outputs_train['s_t_pred_warped_mask'], outputs_train['s_t_1_pred_mask'], self.inputs['of_t'], self.h, self.w, 'loss_temporal')
            self.loss['surf'] = self.get_surf_loss(self.inputs['surfs_t'], outputs_train['F_t'], self.inputs['surfs_dim_t'], batch_size)

            self.loss = self.gather_only_applied_loss(self.loss, self.loss_applied)
            self.loss['total'] = tf.add_n([self.coef_placeholders[key] * val for (key, val) in self.loss.items()], name = 'loss_total')
            print(toRed('{}'.format('applied losses: {}'.format([key for (key, val) in self.loss.items()]))))

        self.loss_test = collections.OrderedDict()

        with tf.name_scope('loss'):
            #self.loss_test['identity'] = tl.cost.mean_squared_error(outputs_test['F_t'], identity, is_mean = True, name = 'loss_identity_t')
            self.loss_test['stable'] = self.masked_MSE(outputs_test['s_t_pred'], self.inputs['s_t_gt'], outputs_test['s_t_pred_mask'], 'loss_stable')
            self.loss_test['temporal'] = self.temporal_loss(outputs_test['s_t_pred_warped'], outputs_test['s_t_1_pred'], outputs_test['s_t_pred_warped_mask'], outputs_test['s_t_1_pred_mask'], self.inputs['of_t'], self.h, self.w, 'loss_temporal')
            self.loss_test['surf'] = self.get_surf_loss(self.inputs['surfs_t'], outputs_test['F_t'], self.inputs['surfs_dim_t'], batch_size)

            self.loss_test = self.gather_only_applied_loss(self.loss_test, self.loss_applied)
            self.loss_test['total'] = tf.add_n([self.coef_placeholders[key] * val for (key, val) in self.loss_test.items()], name = 'loss_total')

    def build_pretrain_optim(self, config):
        with tf.name_scope('Optimizer'):
            self.learning_rate = tf.Variable(config.lr_init, trainable = False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim_init = tf.train.AdamOptimizer(self.learning_rate, beta1 = config.beta1).minimize(self.pretrain_loss['total'], var_list = self.pretraining_vars)

    def build_train_optim(self, config):
        with tf.name_scope('Optimizer'):
            self.learning_rate = tf.Variable(config.lr_init, trainable = False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #optim_main = tf.train.AdamOptimizer(self.learning_rate, beta1 = beta1).minimize(self.loss['total'], var_list = self.training_vars)
                optim_main = tf.train.AdamOptimizer(self.learning_rate, beta1 = config.beta1)
                gradients, self.variables = zip(*optim_main.compute_gradients(self.loss['total'], var_list = self.training_vars))
                self.gradients, _ = tf.clip_by_global_norm(gradients, config.grad_norm_clip_val)
                self.optim_main = optim_main.apply_gradients(zip(self.gradients, self.variables))

    def build_pretrain_summary(self):
        self.loss_epoch_init_placeholder, self.scalar_sum_itr_init, self.image_sum_init, self.summary_epoch_init = self.init_pretrain_summary()

    def init_pretrain_summary(self):
        with tf.name_scope('init_loss_itr'):
            loss_itr_sum_main_list_init = []
            for count, key in enumerate(list(self.pretrain_loss.keys())):
                loss_itr_sum_main_list_init.append(tf.summary.scalar('{}_{}'.format(count, key), self.pretrain_loss[key]))
            loss_itr_sum_main_init = tf.summary.merge(loss_itr_sum_main_list_init)

        with tf.variable_scope('init_images'):
            image_sum_list_init = []
            image_sum_list_init.append(tf.summary.image('1_u_t', fix_image_tf(self.inputs_pretrain['u_t'], 1)))
            image_sum_list_init.append(tf.summary.image('2_s_t_pred', fix_image_tf(self.outputs_pretrain['s_t_pred'], 1)))
            image_sum_list_init.append(tf.summary.image('3_u_t_1', fix_image_tf(self.inputs_pretrain['u_t_1'], 1)))
            image_sum_list_init.append(tf.summary.image('4_s_t_1_pred', fix_image_tf(self.outputs_pretrain['s_t_1_pred'], 1)))

            image_sum_init = tf.summary.merge(image_sum_list_init)

        with tf.name_scope('init_loss_epoch'):
            loss_epoch_sum_main_list_init = []
            loss_epoch_init = collections.OrderedDict.fromkeys(list(self.pretrain_loss.keys()), 0.)
            for count, key in enumerate(loss_epoch_init.keys()):
                loss_epoch_init[key] = tf.placeholder('float32', name = key)
                loss_epoch_sum_main_list_init.append(tf.summary.scalar('{}_{}'.format(count, key), loss_epoch_init[key]))
            loss_epoch_sum_main_init = tf.summary.merge(loss_epoch_sum_main_list_init)

        return loss_epoch_init, loss_itr_sum_main_init, image_sum_init, loss_epoch_sum_main_init

    def build_train_summary(self):
        self.loss_epoch_placeholder, self.scalar_sum_itr, self.image_sum, self.summary_epoch = self.init_train_summary()

    def init_train_summary(self):
        with tf.name_scope('loss_itr'):
            loss_itr_sum_main_list = []
            for count, key in enumerate(list(self.loss.keys())):
                loss_itr_sum_main_list.append(tf.summary.scalar('{}_{}'.format(count, key), self.loss[key]))

            loss_itr_sum_main_list.append(tf.summary.scalar("grad_norm", tf.global_norm(self.gradients)))
            for index, grad in enumerate(self.gradients):
                loss_itr_sum_main_list.append(tf.summary.histogram("{}-grad".format(self.variables[index].name), grad))
                loss_itr_sum_main_list.append(tf.summary.histogram("{}-grad_norm".format(self.variables[index].name), tf.global_norm([grad])))
            loss_itr_sum_main = tf.summary.merge(loss_itr_sum_main_list)
            
        image_sum_list = []
        image_sum_list.append(tf.summary.image('1_u_t', fix_image_tf(self.inputs['u_t'], 1)))
        image_sum_list.append(tf.summary.image('2_s_t_pred', fix_image_tf(self.outputs_train['s_t_pred'], 1)))
        image_sum_list.append(tf.summary.image('3_s_t_gt', fix_image_tf(self.inputs['s_t_gt'], 1)))
        image_sum_list.append(tf.summary.image('4_s_t_prev', fix_image_tf(self.outputs_train['patches_masked'][:, :, :, 3 * (2 * self.sample_num - 3) : 3 * (2 * self.sample_num - 2)], 1)))

        image_sum_list.append(tf.summary.image('5_s_t_1', fix_image_tf(self.outputs_train['s_t_1'], 1)))
        image_sum_list.append(tf.summary.image('6_s_t_1_prev', fix_image_tf(self.outputs_train['patches_masked'][:, :, :, 3 * (2 * self.sample_num - 4) : 3 * (2 * self.sample_num - 3)], 1)))

        image_sum_list.append(tf.summary.image('7_s_t_pred_mask', fix_image_tf(self.outputs_train['s_t_pred_mask'], 1)))
        image_sum_list.append(tf.summary.image('8_s_t_gt_warped', fix_image_tf(self.outputs_train['s_t_gt_warped'], 1)))
        image_sum_list.append(tf.summary.image('9_s_t_1', fix_image_tf(self.outputs_train['s_t_1'], 1)))
        image_sum_list.append(tf.summary.image('10_s_t_1_mask', fix_image_tf(self.outputs_train['s_t_1_mask'], 1)))
        image_sum = tf.summary.merge(image_sum_list)

        with tf.name_scope('loss_epoch'):
            loss_epoch_sum_main_list = []
            loss_epoch = collections.OrderedDict.fromkeys(self.loss_applied, 0.)
            for count, key in enumerate(list(self.loss.keys())):
                loss_epoch[key] = tf.placeholder('float32', name = key)
                loss_epoch_sum_main_list.append(tf.summary.scalar('{}_{}'.format(count, key), loss_epoch[key]))
            loss_epoch_sum_main = tf.summary.merge(loss_epoch_sum_main_list)

        return loss_epoch, loss_itr_sum_main, image_sum, loss_epoch_sum_main

    def init_vars(self, sess):
        self.network.init_vars(sess)

    def update_learning_rate(self, epoch, lr_init, decay_rate, decay_every, sess):
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = decay_rate ** (epoch // decay_every)
            sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(self.learning_rate, lr_init))

    def masked_MSE(self, pred, gt, mask, name):
        pred_masked = pred * mask
        gt_masked = gt * mask

        MSE = tf.reduce_sum(tf.squared_difference(pred_masked, gt_masked), axis = [1, 2, 3])

        # safe_mask = tf.cast(tf.where(tf.equal(mask, tf.zeros_like(mask)), mask + tf.constant(1e-8), mask), tf.float32)
        # MSE = MSE / tf.reduce_sum(safe_mask, axis = [1, 2, 3])

        MSE = tf.div_no_nan(MSE, tf.reduce_sum(mask, axis = [1, 2, 3]))
        return tf.reduce_mean(MSE, name = name)

    def temporal_loss(self, pred, gt, mask_pred, mask_gt, of, h, w, name):
        pred_warped = tf_warp(pred, of, h, w)
        mask_pred_warped = tf_warp(mask_pred, of, h, w)

        #return tl.cost.mean_squared_error(s_t_pred_warped, s_t_1_pred, is_mean = True)
        return self.masked_MSE(pred_warped, gt, mask_pred_warped * mask_gt, name)

    def shape_preserving_loss(self, grid_points, name):
        def get_sp_term(v_src, v_src_0, v_src_1, v, v_0, v_1, batch_size):
            s = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow((v_src - v_src_1), 2), axis = 3))/tf.sqrt(tf.reduce_sum(tf.pow((v_src_0 - v_src_1), 2), axis = 3)), axis = 3)
            M_rot = tf.reshape(tf.tile([0., 1., -1., 0.], [batch_size]), [batch_size, 2, 2]) # [b, 2, 2]
            v_0_1 = v_0 - v_1 # [b, h, w, 2]
            h, w = v_0_1.get_shape().as_list()[1:3]

            v_0_1 = tf.reshape(v_0_1, [-1, h * w, 2]) # [b, h*w, 2]
            v_0_1 = tf.transpose(v_0_1, [0, 2, 1]) # [b, 2, h*w]
            R_v_0_1 = tf.matmul(M_rot, v_0_1) # [b, 2, h * w]

            R_v_0_1 = tf.transpose(R_v_0_1, [0, 2, 1]) # [b, 2, h*w]
            R_v_0_1 = tf.reshape(R_v_0_1, [-1, h, w, 2])

            sp_term = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(v - v_1 - s * R_v_0_1, 2), axis = 3), axis = [1, 2]), name = name)
            return sp_term

        V = self.network.elastic_transformer.get_abs_theta(grid_points)
        shape = tf.shape(V)
        batch_size = shape[0]
        V_src = self.network.elastic_transformer.get_abs_src_points(batch_size)

        v = V[:, :-1, :-1, :]
        v_0 = tf.stop_gradient(V[:, 1:, 1:, :])
        v_1 = tf.stop_gradient(V[:, 1:, :-1, :])

        v_src = V_src[:, :-1, :-1, :]
        v_src_0 = V_src[:, 1:, 1:, :]
        v_src_1 = V_src[:, 1:, :-1, :]

        sp_term_1 = get_sp_term(v_src, v_src_0, v_src_1, v, v_0, v_1, batch_size)

        v = V[:, :-1, 1:, :]
        v_0 = tf.stop_gradient(V[:, 1:, :-1, :])
        v_1 = tf.stop_gradient(V[:, 1:, 1:, :])

        v_src = V_src[:, :-1, 1:, :]
        v_src_0 = V_src[:, 1:, :-1, :]
        v_src_1 = V_src[:, 1:, 1:, :]

        sp_term_2 = get_sp_term(v_src, v_src_0, v_src_1, v, v_0, v_1, batch_size)

        v = V[:, 1:, :-1, :]
        v_0 = tf.stop_gradient(V[:, :-1, 1:, :])
        v_1 = tf.stop_gradient(V[:, :-1, :-1, :])

        v_src = V_src[:, 1:, :-1, :]
        v_src_0 = V_src[:, :-1, 1:, :]
        v_src_1 = V_src[:, :-1, :-1, :]

        sp_term_3 = get_sp_term(v_src, v_src_0, v_src_1, v, v_0, v_1, batch_size)

        v = V[:, 1:, 1:, :]
        v_0 = tf.stop_gradient(V[:, :-1, :-1, :])
        v_1 = tf.stop_gradient(V[:, :-1, 1:, :])

        v_src = V_src[:, 1:, 1:, :]
        v_src_0 = V_src[:, :-1, :-1, :]
        v_src_1 = V_src[:, :-1, 1:, :]
        
        sp_term_4 = get_sp_term(v_src, v_src_0, v_src_1, v, v_0, v_1, batch_size)

        return (sp_term_1 + sp_term_2 + sp_term_3 + sp_term_4) / 4.

    def gather_only_applied_loss(self, loss_dict, loss_applied):
        for key in list(loss_dict.keys()):
            if key not in loss_applied:
                del loss_dict[key]

        return loss_dict

    def init_loss_coefficient(self, config):
        self.loss_applied = config.loss_applied
        self.loss_limit = config.loss_limit.copy()
        self.loss_apply_epoch_range = config.loss_apply_epoch_range.copy()
        self.coef_low = config.coef_low.copy()
        self.coef_high = config.coef_high.copy()
        self.coef_container = config.coef_init.copy()
        self.coef_placeholders = collections.OrderedDict()
        for (key, val) in self.coef_container.items():
             self.coef_placeholders[key] = tf.placeholder_with_default(tf.constant(self.coef_container[key], dtype = 'float32'), shape = None, name = 'stable_coef')

    def adjust_loss_coef(self, feed_dict, epoch, errs):
        if feed_dict is not None:
            for (key, val) in self.coef_placeholders.items():
                if key not in self.loss_applied:
                    self.coef_container[key] = 0.0
                elif errs is not None:
                    if errs[key] > self.loss_limit[key]:
                        self.coef_container[key] = min(self.coef_container[key] * 1.1, self.coef_high[key])
                    elif errs[key] < self.loss_limit[key]:
                        self.coef_container[key] = max(self.coef_container[key] * 0.9, self.coef_low[key])
                elif epoch < self.loss_apply_epoch_range[key][0] or epoch >= self.loss_apply_epoch_range[key][1]:
                    self.coef_container[key] = 0.0

        for (key, val) in self.coef_placeholders.items():
            if epoch < self.loss_apply_epoch_range[key][0] or epoch >= self.loss_apply_epoch_range[key][1] or key not in list(self.loss_applied):
                self.coef_container[key] = 0.0
                feed_dict[val] = self.coef_container[key]

        return feed_dict

    def get_surf_loss(self, surf, H_flat, max_dim_per_batch, batch_size):
        stab_ind = 1
        unstab_ind = 0

        H = tf.reshape(tf.concat([H_flat, tf.ones([batch_size, 1])], axis = 1), [-1, 3, 3])

        surf_dim = tf.shape(surf)[2]
        surf_stab = tf.transpose(tf.concat([surf[:,stab_ind,:,:],tf.ones([batch_size, surf_dim, 1])], 2), [0, 2, 1])

        index_stab = tf.matmul(H, surf_stab)
        z = index_stab[:, 2, :]
        index_stab = tf.concat([tf.expand_dims(tf.div_no_nan(index_stab[:, 0, :], z), 2), tf.expand_dims(tf.div_no_nan(index_stab[:, 1,:], z),2)],2)

        index_stab = index_stab
        index_unstab = surf[:, unstab_ind, :, :]

        # norm_param = tf.constant([w, h], shape = [1, 2], dtype = tf.float32)
        MSE = tf.reduce_sum(tf.squared_difference(index_stab, index_unstab), axis = [1, 2])
        #MSE = tf.reduce_sum(tf.abs(index_stab - index_unstab), axis = [1, 2])
        MSE = tf.div_no_nan(MSE, max_dim_per_batch * 2)

        return tf.reduce_mean(MSE)

