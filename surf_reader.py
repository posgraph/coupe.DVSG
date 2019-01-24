import numpy as np
import tensorflow as tf
import tensorlayer as tl

def read_surf(file_path):
    rawdata = np.loadtxt(file_path)
    output = np.zeros((2,rawdata.shape[0],2))
    for i in range(rawdata.shape[0]):
        output[0,i,0] = rawdata[i,0]
        output[0,i,1] = rawdata[i,1]
        output[1,i,0] = rawdata[i,2]
        output[1,i,1] = rawdata[i,3]

    return np.expand_dims(output, axis = 0)

# surf: [batch, unstab(0)/stab(1), features, value]
def get_surf_loss(surf, H_flat, max_dim_per_batch, batch_size):
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
