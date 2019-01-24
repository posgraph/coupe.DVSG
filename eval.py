import tensorflow as tf
import tensorlayer as tl

import datetime
import time
import os
import collections
import random
import json
import cv2
from threading import Thread
from shutil import rmtree
import numpy as np

from utils import *
from model import StabNet
from trainer import Trainer
from data_loader import Data_Loader
from ckpt_manager import CKPT_Manager

import tensorflow as tf
import tensorlayer as tl

import collections
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

from warp_with_optical_flow import *
from spatial_transformer import *
from networks import *

def get_stable_path_init(stabNet, sample_num):
    STN = ProjectiveTransformer([stabNet.h, stabNet.w])
    stabNet.inputs['IS'] = tf.placeholder('float32', [None, sample_num - 1, stabNet.h, stabNet.w, stabNet.c], name = 'input_S')

    outputs = collections.OrderedDict()
    outputs['src_IS_seq_flat'] = stabNet.flatten_seq(stabNet.inputs['IS'][:, 1:, :, :, :]) # t
    outputs['trg_IS_seq_flat'] = stabNet.flatten_seq(stabNet.inputs['IS'][:, :-1, :, :, :]) # t_1
    outputs['IS_concat'] = tf.concat([outputs['src_IS_seq_flat'], outputs['trg_IS_seq_flat']], axis = 3) # t_1, t
    sample_num_S = sample_num - 2

    with tf.variable_scope('stabNet'):
        outputs['S_t_1_seq_flat_H'] = pathFinder(outputs['IS_concat'], stabNet.F_dim, False, stabNet.get_reuse('pathFinder'), scope = 'pathFinder')
        outputs['S_t_1_seq_flat'] = STN.H2OF(tf.ones_like(outputs['src_IS_seq_flat']), outputs['S_t_1_seq_flat_H'])
        outputs['S_t_1_seq'] = stabNet.unflatten_seq(outputs['S_t_1_seq_flat'], sample_num_S)

    return outputs

def get_unstable_path_init(stabNet, mple_num):
    STN = ProjectiveTransformer([stabNet.h, stabNet.w])
    stabNet.inputs['IU'] = tf.placeholder('float32', [None, sample_num - 1, stabNet.h, stabNet.w, stabNet.c], name = 'input_U')

    outputs = collections.OrderedDict()
    outputs['src_IU_seq_flat'] = stabNet.flatten_seq(stabNet.inputs['IU'][:, 1:, :, :, :])
    outputs['trg_IU_seq_flat'] = stabNet.flatten_seq(stabNet.inputs['IU'][:, :-1, :, :, :])
    outputs['IU_concat'] = tf.concat([outputs['src_IU_seq_flat'], outputs['trg_IU_seq_flat']], axis = 3)
    sample_num_U = sample_num - 2

    with tf.variable_scope('stabNet'):
        outputs['U_t_1_seq_flat_H'] = pathFinder(outputs['IU_concat'], stabNet.F_dim, False, stabNet.get_reuse('pathFinder'), scope = 'pathFinder')
        outputs['U_t_1_seq_flat'] = STN.H2OF(tf.ones_like(outputs['src_IU_seq_flat']), outputs['U_t_1_seq_flat_H'])
        outputs['U_t_1_seq'] = stabNet.unflatten_seq(outputs['U_t_1_seq_flat'], sample_num_U)

    return outputs

def init_evaluation_model(stabNet, sample_num):
    outputs = collections.OrderedDict()
    with tf.variable_scope('stabNet'):
        STN = ProjectiveTransformer([stabNet.h, stabNet.w])
        outputs = collections.OrderedDict()
        stabNet.inputs['Iu'] = tf.placeholder('float32', [None, 1, stabNet.h, stabNet.w, stabNet.c], name = 'input_U')
        stabNet.inputs['U_t_1_seq'] = tf.placeholder('float32', [None, sample_num - 2, stabNet.h, stabNet.w, 2], name = 'input_U_t_1_seq')
        stabNet.inputs['S_t_1_seq'] = tf.placeholder('float32', [None, sample_num - 2, stabNet.h, stabNet.w, 2], name = 'input_S_t_1_seq')
        stabNet.inputs['B_t_1'] = tf.placeholder('float32', [None, stabNet.h, stabNet.w, 2], name = 'input_B_t_1')
        stabNet.inputs['B_t_1_H'] = tf.placeholder('float32', [None, stabNet.h, stabNet.w, 2], name = 'input_B_t_1_H')

        # find path for Iuu
        outputs['Iuu'] = tf.concat([tf.expand_dims(stabNet.inputs['IU'][:, -1, :, :, :], axis = 1), stabNet.inputs['Iu']], axis = 1)
        outputs['src_Iuu_seq_flat'] = stabNet.flatten_seq(outputs['Iuu'][:, 1:, :, :, :]) # t
        outputs['trg_Iuu_seq_flat'] = stabNet.flatten_seq(outputs['Iuu'][:, :-1, :, :, :]) # t_1
        outputs['Iuu_concat'] = tf.concat([outputs['src_Iuu_seq_flat'], outputs['trg_Iuu_seq_flat']], axis = 3) # t_1, t
        sample_num_uu = 1
        outputs['U_t_seq_flat_H'] = pathFinder(outputs['Iuu_concat'], stabNet.F_dim, False, stabNet.get_reuse('pathFinder'), scope = 'pathFinder')
        outputs['U_t_seq_flat'] = STN.H2OF(tf.ones_like(outputs['src_Iuu_seq_flat']), outputs['U_t_seq_flat_H'])
        outputs['U_t_seq'] = stabNet.unflatten_seq(outputs['U_t_seq_flat'], sample_num_uu)
        outputs['U_t'] = outputs['U_t_seq'][:, 0, :, :, :]

        ## 2. P_t_1
        outputs['P_t_1_seq'] = tf.cumsum(stabNet.inputs['S_t_1_seq'], axis = 1)
        outputs['P_t_1'] = outputs['P_t_1_seq'][:, -1, :, :, :]
        ## 3. C_t_1
        outputs['C_t_1_seq'] = tf.cumsum(stabNet.inputs['U_t_1_seq'], axis = 1)
        outputs['C_t_1'] = outputs['C_t_1_seq'][:, -1, :, :, :]

        outputs['B_t_1_cumsum'] = outputs['P_t_1'] - outputs['C_t_1']

        ####################
        ## PATH SMOOTHING ##
        ####################
        with tf.variable_scope('pathSmoother'):
            outputs['U_t_seq_c'] = stabNet.seq_to_channel(tf.concat([stabNet.inputs['U_t_1_seq'], outputs['U_t_seq']], axis = 1))
            outputs['S_t_1_seq_c'] = stabNet.seq_to_channel(stabNet.inputs['S_t_1_seq'])
            outputs['S_t_pred_H'] = pathPredictor(tf.concat([outputs['S_t_1_seq_c'], outputs['U_t_seq_c']], axis = 3), stabNet.F_dim, False, stabNet.get_reuse('pathPredictor'), scope = 'pathPredictor')
            outputs['S_t_pred'] = STN.H2OF(tf.ones_like(stabNet.inputs['Iu'][:, 0, :, :, :]), outputs['S_t_pred_H'])
            outputs['S_t_pred_seq'] = tf.expand_dims(outputs['S_t_pred'], axis = 1)

            outputs['IUu_seq_c'] = stabNet.seq_to_channel(tf.concat([tf.expand_dims(stabNet.inputs['IU'][:, -1, :, :, :], axis = 1), stabNet.inputs['Iu']], axis = 1))
            outputs['B_t_pred_H'] = pathUpdater(tf.concat([tf.stop_gradient(outputs['S_t_pred']), stabNet.inputs['B_t_1'], outputs['U_t'], outputs['IUu_seq_c']], axis = 3), stabNet.F_dim, False, stabNet.get_reuse('pathUpdater'), scope = 'pathRefiner')
            outputs['B_t'] = STN.H2OF(tf.ones_like(stabNet.inputs['Iu'][:, 0, :, :, :]), outputs['B_t_pred_H'])
            outputs['B_t_H'] = -1*(outputs['S_t_pred'] - outputs['U_t'])

        #############
        ## WARPING ##
        #############
        # B_t = S_t - U_t + B_t_1 = S_t - U_t + (P_t_1 - C_t_1) - C_0
        outputs['B_t_cumsum'] = -1 * (outputs['S_t_pred'] - outputs['U_t'] + outputs['B_t_1_cumsum'])
        outputs['Iu'] = tf.reshape(stabNet.inputs['Iu'], [-1, stabNet.h, stabNet.w, stabNet.c])
        with tf.variable_scope('STN'):
            outputs['Is_pred'] = tf_warp(outputs['Iu'], outputs['B_t'])
            outputs['Is_pred_H'] = tf_warp(outputs['Iu'], outputs['B_t_H'])
            outputs['Is_pred_cumsum'] = tf_warp(outputs['Iu'], outputs['B_t_cumsum'])

    return outputs

def evaluate(config, mode):
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    save_path = os.path.join(config.LOG_DIR.save, date, config.eval_mode)
    exists_or_mkdir(save_path)

    print(toGreen('Loading checkpoint manager...'))
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, 10)
    #ckpt_manager = CKPT_Manager(config.PRETRAIN.LOG_DIR.ckpt, mode, 10)

    batch_size = config.batch_size
    sample_num = config.sample_num
    skip_length = np.array(config.skip_length)

    ## DEFINE SESSION
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    ## DEFINE MODEL
    print(toGreen('Building model...'))
    stabNet = StabNet(config.height, config.width, config.F_dim, is_train = False)
    stable_path_net = get_stable_path_init(stabNet, sample_num)
    unstable_path_net = get_unstable_path_init(stabNet, sample_num)
    outputs_net = init_evaluation_model(stabNet, sample_num)

    ## INITIALIZING VARIABLE
    print(toGreen('Initializing variables'))
    sess.run(tf.global_variables_initializer())
    print(toGreen('Loading checkpoint...'))
    ckpt_manager.load_ckpt(sess, by_score = config.load_ckpt_by_score)

    print(toYellow('======== EVALUATION START ========='))
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.unstab_path, regx = '.*', printable = False)))
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        cap = cv2.VideoCapture(os.path.join(config.unstab_path, test_video_name))
        fps = cap.get(5)
        resize_h = config.height
        resize_w = config.width
        # out_h = int(cap.get(4))
        # out_w = int(cap.get(3))
        out_h = resize_h
        out_w = resize_w

        # refine_temp = np.ones((h, w))
        # refine_temp = refine_image(refine_temp)
        # [h, w] = refine_temp.shape[:2]
        total_frame_num = int(cap.get(7))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(os.path.join(save_path, str(k) + '_' + config.eval_mode + '_' + base_name + '_out.avi'), fourcc, fps, (4 * out_w, out_h))
        print(toYellow('reading filename: {}, total frame: {}'.format(test_video_name, total_frame_num)))

        # read frame
        def refine_frame(frame):
            return cv2.resize(frame / 255., (resize_w, resize_h))

        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (out_w, out_h))
            return ref, frame

        # reading all frames in the video
        total_frames = []
        print(toGreen('reading all frames...'))
        while True:
            ref, frame = read_frame(cap)
            if ref == False:
                break
            total_frames.append(refine_frame(frame))

        # duplicate first frames 30 times
        for i in np.arange(skip_length[-1] - skip_length[0]):
            total_frames.insert(0, total_frames[0])

        print(toGreen('stabilizaing video...'))
        total_frame_num = len(total_frames)
        total_frames = np.array(total_frames)

        S = [None] * total_frame_num
        U = [None] * total_frame_num
        B_t_1_list = [None] * total_frame_num
        B_t_1_H_list = [None] * total_frame_num
        Is_t_1_list = [None] * total_frame_num
        S_t_1_seq = None
        U_t_1_seq = None

        for i in np.arange(skip_length[-1] - skip_length[0]):
            B_t_1_list[i] = np.zeros([1, out_h, out_w, 2])
            B_t_1_H_list[i] = np.zeros([1, out_h, out_w, 2])

        sample_idx = skip_length
        for frame_idx in range(skip_length[-1] - skip_length[0], total_frame_num):

            batch_frames = total_frames[sample_idx]
            batch_frames = np.expand_dims(np.concatenate(np.expand_dims(batch_frames, axis = 0), axis = 0), axis = 0)

            if U[sample_idx[0]] is None:
                feed_dict = {stabNet.inputs['IS']: batch_frames[:, :-1, :, :, :]}
                stable_path_init = sess.run(stable_path_net, feed_dict)

                feed_dict = {stabNet.inputs['IU']: batch_frames[:, :-1, :, :, :]}
                unstable_path_init = sess.run(unstable_path_net, feed_dict)

                S_t_1_seq = stable_path_init['S_t_1_seq']
                U_t_1_seq = unstable_path_init['U_t_1_seq']

                for i in np.arange(S_t_1_seq.shape[1]):
                    S[sample_idx[i + 1]] = S_t_1_seq[:, i:i+1, :, :, :]
                    U[sample_idx[i + 1]] = U_t_1_seq[:, i:i+1, :, :, :]

            idxs = sample_idx[1:-1]
            i = 0
            for idx in idxs:
                if i == 0:
                    S_t_1_seq = S[idx]
                    U_t_1_seq = U[idx]
                else:
                    S_t_1_seq = np.concatenate([S_t_1_seq, S[idx]], axis = 1)
                    U_t_1_seq = np.concatenate([U_t_1_seq, U[idx]], axis = 1)
                i += 1
            B_t_1 = B_t_1_list[sample_idx[-2]]
            B_t_1_H = B_t_1_H_list[sample_idx[-2]]

            feed_dict = {
                stabNet.inputs['IU']:batch_frames[:, :-1, :, :, :],
                stabNet.inputs['Iu']: np.expand_dims(batch_frames[:, -1, :, :, :], axis = 1),
                stabNet.inputs['U_t_1_seq']: U_t_1_seq,
                stabNet.inputs['S_t_1_seq']: S_t_1_seq, 
                stabNet.inputs['B_t_1']: B_t_1, 
                stabNet.inputs['B_t_1_H']: B_t_1_H, 
            }
            Is_pred, Is_pred_H, Is_pred_cumsum, S_t_pred, U_t, B_t_1, B_t_1_H = sess.run([outputs_net['Is_pred'], outputs_net['Is_pred_H'], outputs_net['Is_pred_cumsum'], outputs_net['S_t_pred_seq'], outputs_net['U_t_seq'], outputs_net['B_t'], outputs_net['B_t_H']], feed_dict)

            S[sample_idx[-1]] = S_t_pred
            U[sample_idx[-1]] = U_t
            B_t_1_list[sample_idx[-1]] = B_t_1
            B_t_1_H_list[sample_idx[-1]] = B_t_1_H

            Is_pred = np.squeeze(Is_pred)
            Is_pred_H = np.squeeze(Is_pred_H)
            Is_pred_cumsum = np.squeeze(Is_pred_cumsum)

            output = np.uint8(np.concatenate((total_frames[frame_idx].copy(), Is_pred, Is_pred_H, Is_pred_cumsum), axis = 1) * 255.)
            #output = np.uint8(np.concatenate((total_frames[frame_idx].copy(), Is_pred), axis = 1) * 255.)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(np.uint8(output))

            print('{}/{} {}/{} frame index: {}'.format(k + 1, len(test_video_list), frame_idx, int(total_frame_num - 1), sample_idx), flush = True)
            sample_idx = sample_idx + 1


        cap.release()
        out.release()

def handle_directory(config, delete_log):
    def mkdir(dir_dict, delete_log_, delete_ckpt_ = True):
        for (key, val) in dir_dict.items():
            if 'perm' in key and delete_ckpt_ is False:
                exists_or_mkdir(val)
                continue

            if delete_log_:
                rmtree(val, ignore_errors = True)
            exists_or_mkdir(val)

    delete_log = delete_log

    if delete_log:
        delete_log = input('Are you sure to delete the logs (y/n): ')

        if len(delete_log) == 0 or delete_log[0].lower() == 'y':
            delete_log = True
        elif delete_log[0].lower() == 'n':
            delete_log = False
        else:
            print('invalid input')
            exit()

    if 'is_pretrain' in list(config.keys()) and config.is_pretrain:
        delete_ckpt = True if config.PRETRAIN.delete_log else False
        mkdir(config.PRETRAIN.LOG_DIR, delete_log, delete_ckpt)
    mkdir(config.LOG_DIR, delete_log)

if __name__ == '__main__':
    import argparse
    from config import get_config, log_config, print_config

    parser = argparse.ArgumentParser()
    config_init = get_config()
    parser.add_argument('-m', '--mode', type = str, default = 'poseNet', help = 'model name')
    parser.add_argument('-dl', '--delete_log', type = str , default = 'false', help = 'whether to delete log or not')

    parser.add_argument('-t', '--is_train', type = str , default = 'true', help = 'whether to train or not')

    parser.add_argument('-em', '--eval_mode', type=str, default = 'eval', help = 'limits of losses that controls coefficients')
    parser.add_argument('-esk', '--eval_skip_length', type=int, default = config_init.TRAIN.skip_length, help = 'limits of losses that controls coefficients')

    parser.add_argument('-ckpt_sc', '--load_ckpt_by_score', type=str, default = config_init.EVAL.load_ckpt_by_score, help = 'limits of losses that controls coefficients')
    args = parser.parse_args()

    config = get_config(args.mode)
    config.is_train = t_or_f(args.is_train)
    config.delete_log = t_or_f(args.delete_log)

    config.EVAL.skip_length = args.eval_skip_length
    config.EVAL.load_ckpt_by_score = t_or_f(args.load_ckpt_by_score)
    config.EVAL.eval_mode = args.eval_mode
    print(toWhite('Creating log directories...'))
    handle_directory(config.EVAL, False)

    tl.logging.set_verbosity(tl.logging.DEBUG)
    #tl.logging.set_verbosity(tl.logging.INFO)

    print(toYellow('\n[TESTING {}]\n'.format(config.mode)))
    evaluate(config.EVAL, config.mode)
