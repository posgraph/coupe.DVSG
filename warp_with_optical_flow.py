import tensorflow as tf

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def tf_warp_prev(img, flow, H, W):
    flow = tf.transpose(flow, [0, 3, 1, 2])
    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,0)

    y = tf.expand_dims(y,0)
    y = tf.expand_dims(y,0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    grid  = tf.concat([x,y],axis = 1)
    flows = grid+flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:,0,:,:]
    y = flows[:,1,:,:]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0,  tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)


    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.tile(tf.expand_dims(x,1), [1, n_repeats])
        return tf.reshape(rep, [-1])

def tf_warp(im, flow, out_height, out_width):
    size = tf.shape(im)
    batch_size = size[0]
    height = size[1]
    width = size[2]
    num_channels = size[3]

    edge_size = 1
    im = tf.pad(im, [[0,0], [edge_size,edge_size], [edge_size,edge_size], [0,0]], mode='CONSTANT')

    flow = tf.transpose(flow, [0, 3, 1, 2])
    x,y = tf.meshgrid(tf.range(width), tf.range(height))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,0)

    y = tf.expand_dims(y,0)
    y = tf.expand_dims(y,0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    grid  = tf.concat([x,y],axis = 1)
    flows = grid+flow
    x = flows[:,0,:,:]
    y = flows[:,1,:,:]

    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    x = tf.clip_by_value(x, -edge_size, width_f - 1 + edge_size)
    y = tf.clip_by_value(y, -edge_size, height_f - 1 + edge_size)

    x += edge_size
    y += edge_size

    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    x1 = tf.cast(tf.minimum(x1_f, width_f - 1 + 2 * edge_size),  tf.int32)
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1 + 2 * edge_size), tf.int32)

    dim2 = width + 2 * edge_size
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)

    base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2

    idx_00 = base_y0 + x0
    idx_01 = base_y0 + x1
    idx_10 = base_y1 + x0
    idx_11 = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, [-1, num_channels])

    I00 = tf.gather(im_flat, idx_00)
    I01 = tf.gather(im_flat, idx_01)
    I10 = tf.gather(im_flat, idx_10)
    I11 = tf.gather(im_flat, idx_11)

    # and finally calculate interpolated values
    w00 = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
    w01 = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
    w10 = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
    w11 = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)

    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    _, _, _, num_channels = im.get_shape().as_list()
    output = tf.reshape(output, [-1, out_height, out_width, num_channels])
    return output
