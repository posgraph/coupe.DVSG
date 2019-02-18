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

def train(config, mode):
    ## Managers
    print(toGreen('Loading checkpoint manager...'))
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, mode, config.max_ckpt_num)
    ckpt_manager_itr = CKPT_Manager(config.LOG_DIR.ckpt_itr, mode, config.max_ckpt_num)
    ckpt_manager_init = CKPT_Manager(config.PRETRAIN.LOG_DIR.ckpt, mode, config.max_ckpt_num)
    ckpt_manager_init_itr = CKPT_Manager(config.PRETRAIN.LOG_DIR.ckpt_itr, mode, config.max_ckpt_num)
    ckpt_manager_perm = CKPT_Manager(config.PRETRAIN.LOG_DIR.ckpt_perm, mode, 1)

    ## DEFINE SESSION
    seed_value = 1
    tf.set_random_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    print(toGreen('Initializing session...'))
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    ## DEFINE MODEL
    stabNet = StabNet(config.height, config.width)

    ## DEFINE DATA LOADERS
    print(toGreen('Loading dataloader...'))
    data_loader = Data_Loader(config, is_train = True, thread_num = config.thread_num)
    data_loader_test = Data_Loader(config.TEST, is_train = False, thread_num = config.thread_num)

    ## DEFINE TRAINER
    print(toGreen('Initializing Trainer...'))
    trainer = Trainer(stabNet, [data_loader, data_loader_test], config)

    ## DEFINE SUMMARY WRITER
    print(toGreen('Building summary writer...'))
    if config.is_pretrain:
        writer_scalar_itr_init = tf.summary.FileWriter(config.PRETRAIN.LOG_DIR.log_scalar_train_itr, flush_secs = 30, filename_suffix = '.scalor_log_itr_init')
        writer_scalar_epoch_init = tf.summary.FileWriter(config.PRETRAIN.LOG_DIR.log_scalar_train_epoch, flush_secs = 30, filename_suffix = '.scalor_log_epoch_init')
        writer_scalar_epoch_valid_init = tf.summary.FileWriter(config.PRETRAIN.LOG_DIR.log_scalar_valid, flush_secs = 30, filename_suffix = '.scalor_log_epoch_test_init')
        writer_image_init = tf.summary.FileWriter(config.PRETRAIN.LOG_DIR.log_image, flush_secs = 30, filename_suffix = '.image_log_init')

    writer_scalar_itr = tf.summary.FileWriter(config.LOG_DIR.log_scalar_train_itr, flush_secs = 30, filename_suffix = '.scalor_log_itr')
    writer_scalar_epoch = tf.summary.FileWriter(config.LOG_DIR.log_scalar_train_epoch, flush_secs = 30, filename_suffix = '.scalor_log_epoch')
    writer_scalar_epoch_valid = tf.summary.FileWriter(config.LOG_DIR.log_scalar_valid, flush_secs = 30, filename_suffix = '.scalor_log_epoch_test')
    writer_image = tf.summary.FileWriter(config.LOG_DIR.log_image, flush_secs = 30, filename_suffix = '.image_log')

    ## INITIALIZE SESSION
    print(toGreen('Initializing network...'))
    sess.run(tf.global_variables_initializer())
    trainer.init_vars(sess)

    ckpt_manager_init.load_ckpt(sess, by_score = False)
    ckpt_manager_perm.load_ckpt(sess)

    if config.is_pretrain:
        print(toYellow('======== PRETRAINING START ========='))
        global_step = 0
        for epoch in range(0, config.PRETRAIN.n_epoch):
        #for epoch in range(0, 1):

            # update learning rate
            trainer.update_learning_rate(epoch, config.PRETRAIN.lr_init, config.PRETRAIN.lr_decay_rate, config.PRETRAIN.decay_every, sess)

            errs_total_pretrain = collections.OrderedDict.fromkeys(trainer.pretrain_loss.keys(), 0.)
            errs = None
            epoch_time = time.time()
            idx = 0
            while True:
            #for idx in range(0, 2):
                step_time = time.time()

                feed_dict, is_end = data_loader.feed_the_network()
                if is_end: break
                feed_dict = trainer.adjust_loss_coef(feed_dict, epoch, errs)
                _, lr, errs = sess.run([trainer.optim_init, trainer.learning_rate, trainer.pretrain_loss], feed_dict)
                errs_total_pretrain = dict_operations(errs_total_pretrain, '+', errs)

                if global_step % config.write_log_every_itr == 0:
                    summary_loss_itr, summary_image  = sess.run([trainer.scalar_sum_itr_init, trainer.image_sum_init], feed_dict)
                    writer_scalar_itr_init.add_summary(summary_loss_itr, global_step)
                    writer_image_init.add_summary(summary_image, global_step)

                # save checkpoint
                if (global_step) % config.PRETRAIN.write_ckpt_every_itr == 0:
                    ckpt_manager_init_itr.save_ckpt(sess, trainer.pretraining_save_vars, '{:05d}_{:05d}'.format(epoch, global_step), score = errs_total_pretrain['total'] / (idx + 1))

                print_logs('PRETRAIN', mode, epoch, step_time, idx, data_loader.num_itr, errs = errs, coefs = trainer.coef_container, lr = lr)

                global_step += 1
                idx += 1

            # save log
            errs_total_pretrain = dict_operations(errs_total_pretrain, '/', data_loader.num_itr)
            summary_loss_epoch_init = sess.run(trainer.summary_epoch_init, feed_dict = dict_operations(trainer.loss_epoch_init_placeholder, '=', errs_total_pretrain))
            writer_scalar_epoch_init.add_summary(summary_loss_epoch_init, epoch)

            ## TEST
            errs_total_pretrain_test = collections.OrderedDict.fromkeys(trainer.pretrain_loss_test.keys(), 0.)
            errs = None
            epoch_time_test = time.time()
            idx = 0
            while True:
            #for idx in range(0, 2):
                step_time = time.time()

                feed_dict, is_end = data_loader_test.feed_the_network()
                if is_end: break
                feed_dict = trainer.adjust_loss_coef(feed_dict, epoch, errs)
                errs = sess.run(trainer.pretrain_loss_test, feed_dict)

                errs_total_pretrain_test = dict_operations(errs_total_pretrain_test, '+', errs)

                print_logs('PRETRAIN TEST', mode, epoch, step_time, idx, data_loader_test.num_itr, errs = errs, coefs = trainer.coef_container)
                idx += 1

            # save log
            errs_total_pretrain_test = dict_operations(errs_total_pretrain_test, '/', data_loader_test.num_itr)
            summary_loss_test_init = sess.run(trainer.summary_epoch_init, feed_dict = dict_operations(trainer.loss_epoch_init_placeholder, '=', errs_total_pretrain_test))
            writer_scalar_epoch_valid_init.add_summary(summary_loss_test_init, epoch)

            print_logs('TRAIN SUMMARY', mode, epoch, epoch_time, errs = errs_total_pretrain)
            print_logs('TEST SUMMARY', mode, epoch, epoch_time_test, errs = errs_total_pretrain_test)

            # save checkpoint
            if epoch % config.write_ckpt_every_epoch == 0:
                ckpt_manager_init.save_ckpt(sess, trainer.pretraining_save_vars, epoch, score = errs_total_pretrain_test['total'])

            # reset image log
            if epoch % config.refresh_image_log_every_epoch == 0:
                writer_image_init.close()
                remove_file_end_with(config.PRETRAIN.LOG_DIR.log_image, '*.image_log')
                writer_image_init.reopen()

        if config.pretrain_only:
            return
        else:
            data_loader.reset_to_train_input(stabNet)
            data_loader_test.reset_to_train_input(stabNet)

    print(toYellow('========== TRAINING START =========='))
    global_step = 0
    for epoch in range(0, config.n_epoch):
    #for epoch in range(0, 1):
        # update learning rate
        trainer.update_learning_rate(epoch, config.lr_init, config.lr_decay_rate, config.decay_every, sess)

        ## TRAIN
        errs_total_train = collections.OrderedDict.fromkeys(trainer.loss.keys(), 0.)
        errs = None
        epoch_time = time.time()
        idx = 0
        #while True:
        for idx in range(0, 2):
            step_time = time.time()

            feed_dict, is_end = data_loader.feed_the_network()
            if is_end: break
            feed_dict = trainer.adjust_loss_coef(feed_dict, epoch, errs)
            _, lr, errs = sess.run([trainer.optim_main, trainer.learning_rate, trainer.loss], feed_dict)

            errs_total_train = dict_operations(errs_total_train, '+', errs)

            if global_step % config.write_ckpt_every_itr == 0:
                ckpt_manager_itr.save_ckpt(sess, trainer.save_vars, '{:05d}_{:05d}'.format(epoch, global_step), score = errs_total_train['total'] / (idx + 1))

            if global_step % config.write_log_every_itr == 0:
                summary_loss_itr, summary_image  = sess.run([trainer.scalar_sum_itr, trainer.image_sum], feed_dict)
                writer_scalar_itr.add_summary(summary_loss_itr, global_step)
                writer_image.add_summary(summary_image, global_step)

            print_logs('TRAIN', mode, epoch, step_time, idx, data_loader.num_itr, errs = errs, coefs = trainer.coef_container, lr = lr)
            global_step += 1
            idx += 1

        # SAVE LOGS
        errs_total_train = dict_operations(errs_total_train, '/', data_loader.num_itr)
        summary_loss_epoch = sess.run(trainer.summary_epoch, feed_dict = dict_operations(trainer.loss_epoch_placeholder, '=', errs_total_train))
        writer_scalar_epoch.add_summary(summary_loss_epoch, epoch)

        ## TEST
        errs_total_test = collections.OrderedDict.fromkeys(trainer.loss_test.keys(), 0.)
        epoch_time_test = time.time()
        idx = 0
        #while True:
        for idx in range(0, 2):
            step_time = time.time()

            feed_dict, is_end = data_loader_test.feed_the_network()
            if is_end: break
            feed_dict = trainer.adjust_loss_coef(feed_dict, epoch, errs)
            errs = sess.run(trainer.loss_test, feed_dict)

            errs_total_test = dict_operations(errs_total_test, '+', errs)
            print_logs('TEST', mode, epoch, step_time, idx, data_loader_test.num_itr, errs = errs, coefs = trainer.coef_container)
            idx += 1

        # SAVE LOGS
        errs_total_test = dict_operations(errs_total_test, '/', data_loader_test.num_itr)
        summary_loss_epoch_test = sess.run(trainer.summary_epoch, feed_dict = dict_operations(trainer.loss_epoch_placeholder, '=', errs_total_test))
        writer_scalar_epoch_valid.add_summary(summary_loss_epoch_test, epoch)

        ## CKPT
        if epoch % config.write_ckpt_every_epoch == 0:
            ckpt_manager.save_ckpt(sess, trainer.save_vars, epoch, score = errs_total_test['total'])

        ## RESET IMAGE SUMMARY
        if epoch % config.refresh_image_log_every_epoch == 0:
            writer_image.close()
            remove_file_end_with(config.LOG_DIR.log_image, '*.image_log')
            writer_image.reopen()

        print_logs('TRAIN SUMMARY', mode, epoch, epoch_time, errs = errs_total_train)
        print_logs('TEST SUMMARY', mode, epoch, epoch_time_test, errs = errs_total_test)

def evaluate(config, mode):
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    save_path = os.path.join(config.LOG_DIR.save, date, config.eval_mode)
    exists_or_mkdir(save_path)

    print(toGreen('Loading checkpoint manager...'))
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, 10)
    #ckpt_manager = CKPT_Manager(config.PRETRAIN.LOG_DIR.ckpt, mode, 10)

    batch_size = config.batch_size
    sample_num = config.sample_num
    skip_length = config.skip_length

    ## DEFINE SESSION
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    ## DEFINE MODEL
    print(toGreen('Building model...'))
    stabNet = StabNet(config.height, config.width, config.F_dim, is_train = False)
    stable_path_net = stabNet.get_stable_path_init(sample_num)
    unstable_path_net = stabNet.get_unstable_path_init(sample_num)
    outputs_net = stabNet.init_evaluation_model(sample_num)

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

        out = cv2.VideoWriter(os.path.join(save_path, str(k) + '_' + config.eval_mode + '_' + base_name + '_out.avi'), fourcc, fps, (3 * out_w, out_h))
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
        for i in np.arange((sample_num - 1) * skip_length):
            total_frames.insert(0, total_frames[0])

        print(toGreen('stabilizaing video...'))
        total_frame_num = len(total_frames)
        total_frames = np.array(total_frames)

        S = [None] * total_frame_num
        U = [None] * total_frame_num
        C_0_list = [None] * total_frame_num
        S_t_1_seq = None
        U_t_1_seq = None

        for i in np.arange((sample_num - 1) * skip_length):
            C_0_list[i] = np.zeros([1, out_h, out_w, 2])

        sample_idx = np.arange(0, 0 + sample_num * skip_length, skip_length)
        for frame_idx in range((sample_num - 1) * skip_length, total_frame_num):

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
            C_0 = C_0_list[sample_idx[0]]

            feed_dict = {
                stabNet.inputs['IU']:batch_frames[:, :-1, :, :, :],
                stabNet.inputs['Iu']: np.expand_dims(batch_frames[:, -1, :, :, :], axis = 1),
                stabNet.inputs['U_t_1_seq']: U_t_1_seq,
                stabNet.inputs['S_t_1_seq']: S_t_1_seq, 
                stabNet.inputs['C_0']: C_0, 
            }
            Is_pred, Is_pred_wo_C0, S_t_pred, U_t, C_0 = sess.run([outputs_net['Is_pred'], outputs_net['Is_pred_wo_C0'], outputs_net['S_t_pred_seq'], outputs_net['U_t_seq'], outputs_net['B_t_wo_C0']], feed_dict)

            S[sample_idx[-1]] = S_t_pred
            U[sample_idx[-1]] = U_t
            C_0_list[sample_idx[-1]] = C_0

            Is_pred = np.squeeze(Is_pred)
            Is_pred_wo_C0 = np.squeeze(Is_pred_wo_C0)

            output = np.uint8(np.concatenate((total_frames[frame_idx].copy(), Is_pred, Is_pred_wo_C0), axis = 1) * 255.)
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

    parser.add_argument('-b', '--batch_size', type = int, default = config_init.TRAIN.batch_size, help = 'whether to train or not')
    parser.add_argument('-gc', '--grad_norm_clip_val', type = float, default = 5., help = 'gradient norm clipping value')
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-3, help = 'learning_rate')
    parser.add_argument('-sk', '--skip_length', type=int, default = config_init.TRAIN.skip_length, help = 'limits of losses that controls coefficients')

    parser.add_argument('-la',  '--loss_applied', type = str , default = str(config_init.TRAIN.loss_applied), help = 'losses to use')
    parser.add_argument('-lm', '--loss_limit', type=json.loads, default = config_init.TRAIN.loss_limit, help = 'limits of losses that controls coefficients')
    parser.add_argument('-ler', '--loss_apply_epoch_range', type=json.loads, default = config_init.TRAIN.loss_apply_epoch_range, help = 'limits of losses that controls coefficients')
    parser.add_argument('-cl', '--coef_low', type=json.loads, default = None, help = 'minimum coefficient for losses')
    parser.add_argument('-ch', '--coef_high', type=json.loads, default = None, help = 'maximum coefficient for losses')
    parser.add_argument('-ci', '--coef_init', type=json.loads, default = config_init.TRAIN.coef_init, help = 'initial coefficient for losses')

    parser.add_argument('-pt', '--is_pretrain', type = str , default = 'false', help = 'whether to pretrain or not')
    parser.add_argument('-pto', '--pretrain_only', type = str , default = 'false', help = 'whether to pretrain or not')
    parser.add_argument('-pdl', '--pdelete_log', type = str , default = 'false', help = 'whether to delete log or not')
    parser.add_argument('-plr', '--pretrain_learning_rate', type = float, default = None, help = 'learning_rate')

    parser.add_argument('-pla',  '--ploss_applied', type = str , default = None, help = 'losses to use')
    parser.add_argument('-plm', '--ploss_limit', type=json.loads, default = None, help = 'limits of losses that controls coefficients')
    parser.add_argument('-pler', '--ploss_apply_epoch_range', type=json.loads, default = None, help = 'limits of losses that controls coefficients')
    parser.add_argument('-pcl', '--pcoef_low', type=json.loads, default = None, help = 'minimum coefficient for losses')
    parser.add_argument('-pch', '--pcoef_high', type=json.loads, default = None, help = 'maximum coefficient for losses')
    parser.add_argument('-pci', '--pcoef_init', type=json.loads, default = None, help = 'initial coefficient for losses')

    parser.add_argument('-em', '--eval_mode', type=str, default = 'eval', help = 'limits of losses that controls coefficients')
    parser.add_argument('-esk', '--eval_skip_length', type=int, default = config_init.TRAIN.skip_length, help = 'limits of losses that controls coefficients')

    parser.add_argument('-max_ckpt', '--max_ckpt_num', type=int, default = config_init.TRAIN.max_ckpt_num, help = 'number of ckpt to keep')
    parser.add_argument('-ckpt_sc', '--load_ckpt_by_score', type=str, default = config_init.EVAL.load_ckpt_by_score, help = 'limits of losses that controls coefficients')
    args = parser.parse_args()

    config = get_config(args.mode)
    config.is_train = t_or_f(args.is_train)
    config.delete_log = t_or_f(args.delete_log)

    if config.is_train:
        config.TRAIN.is_pretrain = t_or_f(args.is_pretrain)
        config.TRAIN.pretrain_only = t_or_f(args.pretrain_only)
        config.TRAIN.batch_size = args.batch_size
        config.TRAIN.grad_norm_clip_val = args.grad_norm_clip_val
        config.TRAIN.lr_init = args.learning_rate
        config.TRAIN.skip_length = args.skip_length

        config.TRAIN.loss_applied = string_to_array(args.loss_applied)
        config.TRAIN.loss_limit = get_dict_with_list(config.TRAIN.loss_applied, args.loss_limit)
        config.TRAIN.loss_apply_epoch_range = get_dict_with_list(config.TRAIN.loss_applied, args.loss_apply_epoch_range, default_val = config.TRAIN.n_epoch)
        config.TRAIN.coef_init = get_dict_with_list(config.TRAIN.loss_applied, args.coef_init)
        config.TRAIN.coef_low = get_dict_with_list(config.TRAIN.loss_applied, args.coef_low) if args.coef_low is not None else config.TRAIN.coef_init
        config.TRAIN.coef_high = get_dict_with_list(config.TRAIN.loss_applied, args.coef_high) if args.coef_high is not None else config.TRAIN.coef_init

        config.TRAIN.PRETRAIN.delete_log = t_or_f(args.pdelete_log)
        config.TRAIN.PRETRAIN.lr_init = args.pretrain_learning_rate if args.pretrain_learning_rate is not None else args.learning_rate
        config.TRAIN.PRETRAIN.loss_applied = string_to_array(args.ploss_applied) if args.ploss_applied is not None else config.TRAIN.loss_applied
        config.TRAIN.PRETRAIN.loss_limit = get_dict_with_list(config.TRAIN.PRETRAIN.loss_applied, args.ploss_limit) if args.ploss_limit is not None else config.TRAIN.loss_limit
        config.TRAIN.PRETRAIN.loss_apply_epoch_range = get_dict_with_list(config.TRAIN.PRETRAIN.loss_applied, args.ploss_apply_epoch_range, default_val = config.TRAIN.PRETRAIN.n_epoch) if args.ploss_apply_epoch_range is not None else config.TRAIN.loss_apply_epoch_range
        config.TRAIN.PRETRAIN.coef_low = get_dict_with_list(config.TRAIN.PRETRAIN.loss_applied, args.pcoef_low) if args.pcoef_low is not None else config.TRAIN.coef_low
        config.TRAIN.PRETRAIN.coef_high = get_dict_with_list(config.TRAIN.PRETRAIN.loss_applied, args.pcoef_high) if args.pcoef_high is not None else config.TRAIN.coef_high
        config.TRAIN.PRETRAIN.coef_init = get_dict_with_list(config.TRAIN.PRETRAIN.loss_applied, args.pcoef_init) if args.pcoef_init is not None else config.TRAIN.coef_init

        config.TRAIN.max_ckpt_num = args.max_ckpt_num

        print(toWhite('============== CONFIG =============='))
        print_config(config)

        print(toWhite('Creating log directories...'))
        handle_directory(config.TRAIN, config.delete_log)

        print(toWhite('Saving config...'))
        log_config(config.TRAIN.LOG_DIR.config, config)

    else:
        config.EVAL.skip_length = args.eval_skip_length
        config.EVAL.load_ckpt_by_score = t_or_f(args.load_ckpt_by_score)
        config.EVAL.eval_mode = args.eval_mode
        print(toWhite('Creating log directories...'))
        handle_directory(config.EVAL, False)

    tl.logging.set_verbosity(tl.logging.DEBUG)
    #tl.logging.set_verbosity(tl.logging.INFO)
    if config.is_train:
        print(toYellow('\n[TRAINING {}]\n'.format(config.mode)))
        train(config.TRAIN, config.mode)
    else:
        print(toYellow('\n[TESTING {}]\n'.format(config.mode)))
        evaluate(config.EVAL, config.mode)
