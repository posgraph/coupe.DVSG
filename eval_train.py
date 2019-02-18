import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim

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
from trainer import Trainer
from data_loader import Data_Loader
from ckpt_manager import CKPT_Manager
from networks import *
from warp_with_optical_flow import *
from ThinPlateSpline import ThinPlateSpline as stn
from spatial_transformer import *

def get_evaluation_model(sample_num, param_dim, num_control_points, h, w):
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
    outputs['V_src'] = tf.tile(tf.constant(outputs['V_src'].reshape([1, param_dim, 2]), dtype=tf.float32), [tf.shape(inputs['u_t'])[0], 1, 1])


    with tf.variable_scope('stabNet'):
        ## Regressor
        outputs['patches_masked_t'], outputs['random_masks_t'] = random_mask(inputs['patches_t'], [h, w], sample_num)
        with tf.variable_scope('localizationNet') as scope:
            outputs['F_t'] = localizationNet(outputs['patches_masked_t'], param_dim, is_train, reuse = False, scope = scope)

        ## STN
        outputs['s_t_pred'], outputs['x_offset_t'], outputs['y_offset_t'] = stn(inputs['u_t'], outputs['V_src'], outputs['F_t'], [h, w])
        outputs['s_t_pred_mask'], _, _ = stn(tf.ones_like(inputs['u_t']), outputs['V_src'], outputs['F_t'], [h, w])

    return inputs, outputs

def random_mask(patches, out_size, sample_num):
    mask_affine = ProjectiveTransformer(out_size)
    batch_size = tf.shape(patches)[0]

    mask = tf.ones_like(patches[:, :, :, : 3 * (sample_num - 1)])
    H = tf.random_uniform([batch_size, 8], minval = -1, maxval = 1)
    H = H * tf.constant([0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1])
    H = H + tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mask = mask_affine.transform(mask, H)
    mask = tf.concat([mask, tf.ones_like(patches[:, :, :, :3])], axis = 3)

    return patches * mask, mask

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
    num_control_points = 5
    param_dim = num_control_points ** 2
    inputs, outputs = get_evaluation_model(sample_num, param_dim, num_control_points, config.height, config.width)

    ## INITIALIZING VARIABLE
    print(toGreen('Initializing variables'))
    sess.run(tf.global_variables_initializer())
    print(toGreen('Loading checkpoint...'))
    ckpt_manager.load_ckpt(sess, by_score = config.load_ckpt_by_score)

    print(toYellow('======== EVALUATION START ========='))
    offset = '/data1/junyonglee/video_stab/eval'
    file_path = os.path.join(offset, 'train_unstab')
    test_video_list = np.array(sorted(tl.files.load_file_list(path = file_path, regx = '.*', printable = False)))
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        eval_path_stab = os.path.join(offset, 'train_stab')
        eval_path_unstab = os.path.join(offset, 'train_unstab')

        cap_stab = cv2.VideoCapture(os.path.join(eval_path_stab, test_video_name))
        cap_unstab = cv2.VideoCapture(os.path.join(eval_path_unstab, test_video_name))

        total_frame_num = int(cap_unstab.get(7))
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        fps = cap_unstab.get(5)
        out = cv2.VideoWriter(os.path.join(save_path, str(k) + '_' + config.eval_mode + '_' + base_name + '_out.avi'), fourcc, fps, (2 * config.width, config.height))
        print(toYellow('reading filename: {}, total frame: {}'.format(test_video_name, total_frame_num)))

        # read frame
        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame / 255., (config.width, config.height))
            return ref, frame

        # reading all frames in the video
        total_frames_stab = []
        total_frames_unstab = []
        print(toGreen('reading all frames...'))
        while True:
            ref_stab, frame_stab = read_frame(cap_stab)
            ref_unstab, frame_unstab = read_frame(cap_unstab)

            if ref_stab == False or ref_unstab == False:
                break

            total_frames_stab.append(frame_stab)
            total_frames_unstab.append(frame_unstab)

        # duplicate first frames 32 times
        for i in np.arange(skip_length[-1] - skip_length[0]):
            total_frames_unstab[i] = total_frames_stab[i]

        print(toGreen('stabilizaing video...'))
        total_frame_num = len(total_frames_unstab)
        total_frames_stab = np.array(total_frames_stab)
        total_frames_unstab = np.array(total_frames_unstab)

        sample_idx = skip_length
        for frame_idx in range(skip_length[-1] - skip_length[0], total_frame_num):

            batch_frames = total_frames_unstab[sample_idx]
            batch_frames = np.expand_dims(np.concatenate(batch_frames, axis = 2), axis = 0)

            feed_dict = {
                inputs['patches_t']: batch_frames,
                inputs['u_t']: batch_frames[:, :, :, 18:]
            }
            s_t_pred = np.squeeze(sess.run(outputs['s_t_pred'], feed_dict))

            output = np.uint8(np.concatenate([total_frames_unstab[sample_idx[-1]].copy(), s_t_pred], axis = 1) * 255.)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(np.uint8(output))

            total_frames_unstab[sample_idx[-1]] = total_frames_stab[sample_idx[-1]]

            print('{}/{} {}/{} frame index: {}'.format(k + 1, len(test_video_list), frame_idx, int(total_frame_num - 1), sample_idx), flush = True)
            sample_idx = sample_idx + 1

        cap_stab.release()
        cap_unstab.release()
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
    parser.add_argument('-m', '--mode', type = str, default = 'DVGS', help = 'model name')
    parser.add_argument('-dl', '--delete_log', type = str , default = 'false', help = 'whether to delete log or not')

    parser.add_argument('-t', '--is_train', type = str , default = 'False', help = 'whether to train or not')

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
