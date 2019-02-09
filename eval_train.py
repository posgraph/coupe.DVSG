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

from warp_with_optical_flow import *
from ThinPlateSpline import ThinPlateSpline as stn
from spatial_transformer import *
from networks import *

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
    inputs, outputs = StabNet(config.height, config.width).get_evaluation_model(sample_num)

    ## INITIALIZING VARIABLE
    print(toGreen('Initializing variables'))
    sess.run(tf.global_variables_initializer())
    print(toGreen('Loading checkpoint...'))
    ckpt_manager.load_ckpt(sess, by_score = config.load_ckpt_by_score)

    print(toYellow('======== EVALUATION START ========='))
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.unstab_path, regx = '.*', printable = False)))
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        offset = '/data1/junyonglee/video_stab/eval'
        eval_path = os.path.join(offset, 'train')
        cap = cv2.VideoCapture(os.path.join(eval_path, test_video_name))
        fps = cap.get(5)
        out_h = config.height
        out_w = config.width

        total_frame_num = int(cap.get(7))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(os.path.join(save_path, str(k) + '_' + config.eval_mode + '_' + base_name + '_out.avi'), fourcc, fps, (2 * out_w, out_h))
        print(toYellow('reading filename: {}, total frame: {}'.format(test_video_name, total_frame_num)))

        # read frame
        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame / 255., (out_w, out_h))
            return ref, frame

        # reading all frames in the video
        total_frames = []
        print(toGreen('reading all frames...'))
        while True:
            ref, frame = read_frame(cap)
            if ref == False:
                break
            total_frames.append(frame)

        # duplicate first frames 32 times
        for i in np.arange(skip_length[-1] - skip_length[0]):
            total_frames.insert(0, total_frames[0])

        print(toGreen('stabilizaing video...'))
        total_frame_num = len(total_frames)
        total_frames = np.array(total_frames)

        sample_idx = skip_length
        for frame_idx in range(skip_length[-1] - skip_length[0], total_frame_num):

            batch_frames = total_frames[sample_idx]
            batch_frames = np.expand_dims(np.concatenate(batch_frames, axis = 2), axis = 0)

            feed_dict = {
                inputs['patches_t']: batch_frames,
                inputs['u_t']: batch_frames[:, :, :, 18:]
            }
            s_t_pred = np.squeeze(sess.run(outputs['s_t_pred'], feed_dict))

            output = np.uint8(np.concatenate([total_frames[sample_idx[-1]].copy(), s_t_pred], axis = 1) * 255.)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(np.uint8(output))

            total_frames[sample_idx[-1]] = s_t_pred

            if frame_idx == skip_length[-1] - skip_length[0]:
                for i in np.arange(skip_length[-1] - skip_length[0]):
                    total_frames[i] = s_t_pred


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
