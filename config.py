from easydict import EasyDict as edict
import json
import os
import collections

def get_config(mode = ''):
    ## GLOBAL
    config = edict()
    config.project = 'DVSG'
    config.mode = mode
    config.is_train = True
    config.height = 288
    config.width = 512
    # config.height = 224
    # config.width = 400
    config.delete_log = False
    config.thread_num = 4

    ##################################### TRAIN #####################################
    config.TRAIN = edict()
    config.TRAIN.is_pretrain = False
    config.TRAIN.pretrain_only = False
    config.TRAIN.batch_size = 20
    config.TRAIN.n_epoch = 10000
    # learning rate
    config.TRAIN.lr_init = 1e-3
    config.TRAIN.lr_decay_rate = 0.7
    config.TRAIN.decay_every = 5
    # adam
    config.TRAIN.beta1 = 0.9
    # gradient norm
    config.TRAIN.grad_norm_clip_val = 1.0
    # loss coefficients
    config.TRAIN.loss_applied = ['stable', 'border', 'temporal', 'surf']
    config.TRAIN.loss_limit  = 1e-1
    config.TRAIN.loss_apply_epoch_range = [[0, config.TRAIN.n_epoch]]
    config.TRAIN.coef_low = 1.0
    config.TRAIN.coef_high = 1.0
    config.TRAIN.coef_init = 1.0
    # data dir
    offset = '/data1/junyonglee/video_stab/train'
    config.TRAIN.stab_path = os.path.join(offset, 'stab_similarity_frames_origin')
    config.TRAIN.unstab_path = os.path.join(offset, 'unstab_similarity_frames_origin')
    config.TRAIN.of_path = os.path.join(offset, 'optical_flow_s_stabNet')
    config.TRAIN.surf_path = os.path.join(offset, 'surf_stabNet_upgrade')
    # data options
    config.TRAIN.sample_num = 7
    config.TRAIN.skip_length = [0, 16, 24, 28, 30, 31, 32]
    config.TRAIN.height = config.height
    config.TRAIN.width = config.width
    config.TRAIN.thread_num = config.thread_num
    # logs
    config.TRAIN.max_ckpt_num = 10
    config.TRAIN.write_ckpt_every_epoch = 1
    config.TRAIN.refresh_image_log_every_itr = 20
    config.TRAIN.refresh_image_log_every_epoch = 2
    config.TRAIN.write_log_every_itr = 50
    config.TRAIN.write_ckpt_every_itr = 1000
    # log dirs
    config.TRAIN.LOG_DIR = edict()
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(config.mode))
    config.TRAIN.LOG_DIR.ckpt = os.path.join(offset, 'checkpoint', 'train', 'epoch')
    config.TRAIN.LOG_DIR.ckpt_itr = os.path.join(offset, 'checkpoint', 'train', 'itr')
    config.TRAIN.LOG_DIR.log_scalar_train_epoch = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'epoch')
    config.TRAIN.LOG_DIR.log_scalar_train_itr = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'itr')
    config.TRAIN.LOG_DIR.log_scalar_valid = os.path.join(offset, 'log', 'train', 'scalar', 'valid')
    config.TRAIN.LOG_DIR.log_image = os.path.join(offset, 'log', 'train', 'image', 'train')
    config.TRAIN.LOG_DIR.config = os.path.join(offset, 'config')

    #################################### PRETRAIN ###################################
    config.TRAIN.PRETRAIN = edict()
    config.TRAIN.PRETRAIN.n_epoch = 10
    # learning rate
    config.TRAIN.PRETRAIN.lr_init = config.TRAIN.lr_init
    config.TRAIN.PRETRAIN.lr_decay_rate = config.TRAIN.lr_decay_rate
    config.TRAIN.PRETRAIN.decay_every = config.TRAIN.decay_every
    # loss coefficients
    config.TRAIN.PRETRAIN.loss_applied = ['stable', 'border', 'temporal', 'surf']
    config.TRAIN.PRETRAIN.loss_limit  = 1e-1
    config.TRAIN.PRETRAIN.loss_apply_epoch_range = [[0, config.TRAIN.PRETRAIN.n_epoch]]
    config.TRAIN.PRETRAIN.coef_low = 1.0
    config.TRAIN.PRETRAIN.coef_high = 1.0
    config.TRAIN.PRETRAIN.coef_init = 1.0
    # data options
    config.TRAIN.PRETRAIN.sample_num = config.TRAIN.sample_num
    config.TRAIN.PRETRAIN.skip_length = config.TRAIN.skip_length

    # logs
    config.TRAIN.PRETRAIN.write_ckpt_every_itr = 300
    # log dirs
    config.TRAIN.PRETRAIN.LOG_DIR = edict()
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(config.mode))
    config.TRAIN.PRETRAIN.delete_log = False
    config.TRAIN.PRETRAIN.LOG_DIR.ckpt = os.path.join(offset, 'checkpoint', 'pretrain', 'epoch')
    config.TRAIN.PRETRAIN.LOG_DIR.ckpt_itr = os.path.join(offset, 'checkpoint', 'pretrain', 'itr')
    config.TRAIN.PRETRAIN.LOG_DIR.ckpt_perm = os.path.join(offset, 'checkpoint', 'pretrain', 'perm')
    config.TRAIN.PRETRAIN.LOG_DIR.log_scalar_train_epoch = os.path.join(offset, 'log', 'pretrain', 'scalar', 'train', 'epoch')
    config.TRAIN.PRETRAIN.LOG_DIR.log_scalar_train_itr = os.path.join(offset, 'log', 'pretrain', 'scalar', 'train', 'itr')
    config.TRAIN.PRETRAIN.LOG_DIR.log_scalar_valid = os.path.join(offset, 'log', 'pretrain', 'scalar', 'valid')
    config.TRAIN.PRETRAIN.LOG_DIR.log_image = os.path.join(offset, 'log', 'pretrain', 'image', 'train')

    ##################################### TEST ######################################
    config.TRAIN.TEST = edict()
    config.TRAIN.TEST.batch_size = config.TRAIN.batch_size
    # data path
    offset = '/data1/junyonglee/video_stab/test'
    config.TRAIN.TEST.stab_path = os.path.join(offset, 'stab_similarity_frames_origin')
    config.TRAIN.TEST.unstab_path = os.path.join(offset, 'unstab_similarity_frames_origin')
    config.TRAIN.TEST.of_path = os.path.join(offset, 'optical_flow_s_stabNet')
    config.TRAIN.TEST.surf_path = os.path.join(offset, 'surf_stabNet_upgrade')
    # data options
    config.TRAIN.TEST.sample_num = config.TRAIN.sample_num
    config.TRAIN.TEST.skip_length = config.TRAIN.skip_length
    config.TRAIN.TEST.height = config.height
    config.TRAIN.TEST.width = config.width

    ##################################### EVAL ######################################
    config.EVAL = edict()
    config.EVAL.batch_size = 1
    # data path
    offset = '/data1/junyonglee/video_stab/eval'
    config.EVAL.unstab_path = os.path.join(offset, 'unstab')
    # data options
    config.EVAL.sample_num = config.TRAIN.sample_num
    config.EVAL.skip_length = config.TRAIN.skip_length
    config.EVAL.load_ckpt_by_score = True
    config.EVAL.height = config.height
    config.EVAL.width = config.width
    # log dirs
    config.EVAL.LOG_DIR = edict()
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(config.mode))

    config.EVAL.eval_mode = 'eval'
    config.EVAL.LOG_DIR.save = os.path.join(offset, 'result')
    config.EVAL.LOG_DIR.ckpt = config.TRAIN.LOG_DIR.ckpt
    config.EVAL.LOG_DIR.ckpt_itr = config.TRAIN.LOG_DIR.ckpt_itr

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')

def print_config(cfg):
    print(json.dumps(cfg, indent=4))

