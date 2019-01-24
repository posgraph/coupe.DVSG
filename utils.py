import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import numpy as np
import cv2
import math
import operator
import collections
import os
import fnmatch
import termcolor
import time
import string

def load_file_list(root_path):
    folder_paths = []
    file_names = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        if len(dirnames) == 0:
            folder_paths.append(root)
            file_names.append(np.array(sorted(filenames)))
            num_files += len(filenames)

    folder_paths = np.array(folder_paths)
    file_names = np.array(file_names)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    file_names = file_names[sort_idx]

    return folder_paths, file_names, num_files

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass

def refine_image(img):
    h, w = img.shape[:2]

    return img[0 : h - h % 16, 0 : w - w % 16]

def get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))

    return file_path

def remove_file_end_with(path, regex):
    file_paths = get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def fix_image_tf(image, norm_value):
    return tf.cast(image / norm_value * 255., tf.uint8)

def norm_image_tf(image):
    image = image - tf.reduce_min(image, axis = [1, 2, 3], keepdims=True)
    image = image / tf.reduce_max(image, axis = [1, 2, 3], keepdims=True)
    return tf.cast(image * 255., tf.uint8)

def norm_image(image, axis = (1, 2, 3)):
    image = image - np.amin(image, axis = axis, keepdims=True)
    image = image / np.amax(image, axis = axis, keepdims=True)
    return image

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])

def tf_matrix_inverse(matrix):
    a_ = matrix[:, 0, 0]
    b_ = matrix[:, 0, 1]
    c_ = matrix[:, 0, 2]
    d_ = matrix[:, 1, 0]
    e_ = matrix[:, 1, 1]
    f_ = matrix[:, 1, 2]
    g_ = matrix[:, 2, 0]
    h_ = matrix[:, 2, 1]
    i_ = matrix[:, 2, 2]

    matrix_det = a_*e_*i_ + b_*f_*g_ + c_*d_*h_ - c_*e_*g_ - b_*d_*i_ - a_*f_*h_
    matrix_det = tf.reshape(matrix_det, [-1, 1])

    #matrix_adj = tf.linalg.adjoint(matrix)
    adj_a = e_*i_ - f_*h_
    adj_b = -(d_*i_ - g_*f_)
    adj_c = d_*h_ - e_*g_
    adj_d = -(b_*i_ - c_*h_)
    adj_e = a_*i_ - c_*g_
    adj_f = -(a_*h_ - b_*g_)
    adj_g = b_*f_ - c_*e_
    adj_h = -(a_*f_ - c_*d_)
    adj_i = a_*e_ - b_*d_

    r1 = tf.stack([adj_a, adj_b, adj_c], axis = 1)
    r2 = tf.stack([adj_d, adj_e, adj_f], axis = 1)
    r3 = tf.stack([adj_g, adj_h, adj_i], axis = 1)

    matrix_adj_conjugate = tf.stack([r1, r2, r3], axis = 1)
    matrix_adj = tf.matrix_transpose(matrix_adj_conjugate)

    matrix_det_safe = tf.where(tf.equal(matrix_det, tf.zeros_like(matrix_det)), matrix_det + tf.constant(1e-8), matrix_det)
    matrix_inverse = matrix_adj / tf.expand_dims(matrix_det_safe, axis = 2)

    #matrix_inverse = tf.div_no_nan(matrix_adj, tf.expand_dims(matrix_det, axis = 2))

    matrix_inverse = matrix_inverse / tf.expand_dims(tf.expand_dims(matrix_inverse[:, 2, 2], axis = 1), axis = 2)

    return matrix_inverse

def print_logs(train_mode, mode, epoch, time_s, iter = '', iter_total = '', errs = '', coefs = '', lr = None):
    err_str = ''
    if errs != '':
        for key, val in errs.items():
            if key == 'total':
                err_str = '{}: '.format(key) + toRed('{:1.2e}'.format(val)) + err_str
            else:
                if coefs == '':
                    err_str += ', {}: '.format(key) + toBlue('{:1.2e}'.format(val))
                elif key in list(coefs.keys()):
                    err_str += ', {}: '.format(key) + toBlue('{:1.2e}(*{:1.1e})'.format(val, coefs[key]))

        err_str = toWhite('*LOSS->') + '[' + err_str + ']'

    iter_str = ''
    if iter != '':
        iter_str = ' ({}/{})'.format(toCyan('{:04}'.format(iter + 1)), toCyan('{:04}'.format(iter_total)))

    lr_str = ''
    if lr is not None:
        lr_str = ' lr: {}'.format(toGrey('{:1.2e}'.format(lr)))

    print('[{}][{}]{}{}{}{}\n{}\n'.format(
            toWhite(train_mode),
            toYellow(mode),
            toWhite(' {} '.format('EP')) + toCyan('{}'.format(epoch + 1)),
            iter_str,
            lr_str,
            toGreen(' {:5.2f}s'.format(time.time() - time_s)),
            err_str,
            )
        )

def get_dict_with_list(list_key, list_val, default_val = None):

    is_multi_dim = False
    if type(list_val) == list:
        for val in list_val:
            if type(val) == list:
                is_multi_dim = True
                break

    new_dict = collections.OrderedDict()
    for i in np.arange(len(list_key)):
        # epoch range
        if is_multi_dim:
            if len(list_key) == len(list_val):
                list_temp = list_val[i]
            else:
                list_temp = list_val[0]
            if list_temp[1] == -1:
                new_dict[list_key[i]] = [list_temp[0], default_val]
            else:
                new_dict[list_key[i]] = list_temp
        else:
            if type(list_val) is list and len(list_val) == len(list_key):
                new_dict[list_key[i]] = list_val[i]
            else:
                new_dict[list_key[i]] = list_val

    return new_dict

def dict_operations(dict1, op, operand2):
    ops = {'+': operator.add,
           '-': operator.sub,
           '*': operator.mul,
           '/': operator.truediv
           }
    if op != '=':
        if 'dict' in str(type(dict1)).lower() and type(dict1) == type(operand2):
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[key]) for key in dict1.keys()]))
        elif type(operand2) == list:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[count]) for count, key in enumerate(dict1.keys())]))
        elif type(operand2) == int or type(operand2) == float:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2) for key in dict1.keys()]))
    else:
        new_dict = collections.OrderedDict()
        for key in dict1.keys():
            new_dict[dict1[key]] = operand2[key]

        return new_dict

def string_to_array(text):
    x = [word.strip(string.punctuation) for word in text.split()]
    return x

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def load_net_with_different_scope(sess):
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz_dict(name = './pretrained/VS_flownetS2.npz', sess = sess)
    flow_vars = tl.layers.get_variables_with_name('pathFinder', False, False)
    variables_to_resave = []
    for var in flow_vars:
        value = sess.run(var)
        new_name = var.op.name.replace('pathFinder', 'stabNet/pathFinder')
        print('name: ', new_name, ' value: ', value)
        new_var = tf.Variable(value, name = new_name)
        variables_to_resave.append(new_var)

    sess.run(tf.global_variables_initializer())
    tl.files.save_npz_dict(variables_to_resave, name = './flowNetS2_pretrained.npz', sess = sess)
