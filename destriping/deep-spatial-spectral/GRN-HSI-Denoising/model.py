


# external
import numpy as np
import tensorflow as tf


def spatialBlock(input_tensor):
    _, _, _, in_channels = input_tensor.get_shape().as_list()

    channels = in_channels // 2

    theta = tf.keras.layers.Conv2D(channels, 1, padding="valid")(input_tensor)
    theta = tf.reshape(
        theta,
        shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels],
    )

    phi = tf.keras.layers.Conv2D(channels, 1, padding="valid")(input_tensor)
    phi = tf.reshape(
        phi, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels]
    )

    g = tf.keras.layers.Conv2D(channels, 1, padding="valid")(input_tensor)
    g = tf.reshape(
        g, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels]
    )

    phi1 = tf.reshape(phi, shape=[-1, tf.shape(phi)[1] * tf.shape(phi)[2]])
    phi1 = tf.nn.softmax(phi1, axis=-1)
    phi1 = tf.reshape(phi1, shape=[-1, tf.shape(phi)[1], tf.shape(phi)[2]])

    g1 = tf.reshape(g, shape=[-1, tf.shape(g)[1] * tf.shape(g)[2]])
    g1 = tf.nn.softmax(g1, axis=-1)
    g1 = tf.reshape(g1, shape=[-1, tf.shape(g)[1], tf.shape(g)[2]])

    y = tf.matmul(phi1, g1, transpose_a=True)

    #    y = y / tf.cast( tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2],  tf.float32)

    y = tf.matmul(theta, y)

    F_s = tf.reshape(
        y, shape=[-1, tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], channels]
    )

    spatial_out = tf.keras.layers.Conv2D(in_channels, 1, padding="valid")(F_s)

    return spatial_out


def GloRe(X):
    imput_chancel = X.get_shape().as_list()[-1]
    inputs_shape = tf.shape(X)

    N = imput_chancel // 4
    C = imput_chancel // 2

    B = tf.keras.layers.Conv2D(N, 1, padding="valid")(X)
    B = tf.reshape(B, [inputs_shape[0], -1, N])  # [B, H*W, N]

    x_reduced = tf.keras.layers.Conv2D(C, 1, padding="valid")(X)
    x_reduced = tf.reshape(x_reduced, [inputs_shape[0], -1, C])  # [B,  H*W, C]
    x_reduced = tf.transpose(x_reduced, perm=[0, 2, 1])  # [B, C, H*W]

    # [B, C, H * W] * [B, H*W, N] —>#[B, C, N]
    v = tf.matmul(x_reduced, B)  # [B, C, N]

    tmp = tf.reshape(v, shape=[-1, N * C])
    tmp = tf.nn.softmax(tmp, axis=-1)
    v = tf.reshape(tmp, shape=[-1, C, N])

    v = tf.expand_dims(v, axis=1)  # [B, 1, C, N]

    def GCN(Vnode, nodeN, mid_chancel):
        net = tf.keras.layers.Conv2D(N, 1, padding="valid")(Vnode)  # [B, 1, C, N]

        net = Vnode + net  # (I-Ag)V
        net = tf.nn.relu(net)

        net = tf.transpose(net, perm=[0, 3, 1, 2])  # [B, N, 1, C]

        net = tf.keras.layers.Conv2D(mid_chancel, 1, padding="valid")(net)  # [B, N, 1, C]

        return net

    z = GCN(v, N, C)  # [B, N, 1, C]

    #    z = z + tf.transpose(v, perm=[0, 3, 1, 2])

    z = tf.reshape(z, [inputs_shape[0], N, C])  # [B, N, C]

    # [B, H*W, N] * [B, N, C] => [B, H*W, C]
    y = tf.matmul(B, z)  # [B, H*W, C]

    y = tf.expand_dims(y, axis=1)  # [B, 1, H*W, C]
    y = tf.reshape(
        y, [inputs_shape[0], inputs_shape[1], inputs_shape[2], C]
    )  # [B, H, W, C]
    x_res = tf.keras.layers.Conv2D(imput_chancel, 1, padding="valid")(y)

    return x_res


def blockE(_input):
    _, _, _, channels = _input.get_shape().as_list()

    input_tensor = _input

    conv1 = tf.keras.layers.Conv2D(
        channels, 3, padding="SAME", activation=tf.nn.relu
    )(input_tensor)

    #    tmp = tf.concat([input_tensor, conv1],-1)
    tmp = tf.add(input_tensor, conv1)

    conv2 = tf.keras.layers.Conv2D(channels, 3, padding="SAME", activation=tf.nn.relu) (tmp)

    tmp = tf.add(tmp, conv2)
    #    tmp = tf.concat([input_tensor, conv1, conv2],-1)
    conv3 = tf.keras.layers.Conv2D(channels, 3, padding="SAME", activation=tf.nn.relu) (tmp)

    #    tmp = tf.concat([input_tensor, conv1, conv2, conv3],-1)
    tmp = tf.add(tmp, conv3)
    fuse = tf.keras.layers.Conv2D(channels, 1, padding="SAME")(tmp)

    fuse = fuse + GloRe(fuse)

    return fuse + _input


def blockD(_input):
    _, _, _, channels = _input.get_shape().as_list()

    input_tensor = _input + spatialBlock(_input)

    conv1 = tf.keras.layers.Conv2D(
        channels, 3, padding="SAME", activation=tf.nn.relu
    )(input_tensor)

    #    tmp = tf.concat([input_tensor, conv1],-1)
    tmp = tf.add(input_tensor, conv1)
    conv2 = tf.keras.layers.Conv2D(channels, 3, padding="SAME", activation=tf.nn.relu)(tmp)

    #    tmp = tf.concat([input_tensor, conv1, conv2],-1)
    tmp = tf.add(tmp, conv2)
    conv3 = tf.keras.layers.Conv2D(channels, 3, padding="SAME", activation=tf.nn.relu)(tmp)

    #    tmp = tf.concat([input_tensor, conv1, conv2, conv3],-1)
    tmp = tf.add(input_tensor, conv3)
    fuse = tf.keras.layers.Conv2D(channels, 1, padding="SAME")(tmp)

    return fuse + _input


def Inference(images, channels=64):
    inchannels = images.get_shape().as_list()[-1]
    print("Test")
    print("Type of Images: ", type(images))
    print("Shape of Images: ", images.get_shape().as_list())
    with tf.compat.v1.variable_scope("UNet"):

        with tf.compat.v1.variable_scope("basic"):
            basic = tf.keras.layers.Conv2D(channels, 3, padding="SAME")(images)
            basic1 = tf.keras.layers.Conv2D(channels, 3, padding="SAME")(basic)

        with tf.compat.v1.variable_scope("encoder0"):
            encode0 = blockE(basic1)
            donw0 = tf.keras.layers.Conv2D(
                channels, 3, strides=2, padding="SAME", activation=tf.nn.relu
            )(encode0)

        with tf.compat.v1.variable_scope("encoder1"):
            encode1 = blockE(donw0)
            donw1 = tf.keras.layers.Conv2D(
                channels, 3, strides=2, padding="SAME", activation=tf.nn.relu
            )(encode1)

        with tf.compat.v1.variable_scope("encoder2"):
            encode2 = blockE(donw1)
            donw2 = tf.keras.layers.Conv2D(
                channels, 3, strides=2, padding="SAME", activation=tf.nn.relu
            )(encode2)

        with tf.compat.v1.variable_scope("middle"):

            media_end = blockE(donw2)

        with tf.compat.v1.variable_scope("decoder2"):
            Deblock2 = tf.image.resize(media_end, size=[tf.shape(encode2)[1], tf.shape(encode2)[2]], method='bilinear')
            Deblock2 = tf.concat([Deblock2, encode2], -1)
            Deblock2 = tf.keras.layers.Conv2D(
                channels, 1, padding="SAME", activation=tf.nn.relu
            )(Deblock2)
            Deblock2 = blockD(Deblock2)

        with tf.compat.v1.variable_scope("decoder1"):
            Deblock1 = tf.image.resize(Deblock2, [tf.shape(encode1)[1], tf.shape(encode1)[2]], method='bilinear')
            Deblock1 = tf.concat([Deblock1, encode1], -1)
            Deblock1 = tf.keras.layers.Conv2D(
                channels, 1, padding="SAME", activation=tf.nn.relu
            )(Deblock1)
            Deblock1 = blockD(Deblock1)

        with tf.compat.v1.variable_scope("decoder0"):
            Deblock0 = tf.image.resize(Deblock1, [tf.shape(encode1)[1], tf.shape(encode1)[2]], method='bilinear')
            Deblock0 = tf.concat([Deblock0, encode0, basic1], axis=-1)
            Deblock0 = tf.keras.layers.Conv2D(
                channels, 1, padding="SAME", activation=tf.nn.relu
            )(Deblock0)
            Deblock0 = blockD(Deblock0)

        with tf.compat.v1.variable_scope("reconstruct"):
            decoding_end = Deblock0 + basic
            res = tf.keras.layers.Conv2D(inchannels, 3, padding="SAME")(decoding_end)
            out = images + res

    return out


if __name__ == "__main__":
    ##tf.reset_default_graph()
    # change placeholder
    input_x = tf.keras.Input(shape=(101, 101, 1))
    
    #    out = spatialBlock(input_x)
    out = Inference(input_x)

    print(input_x.get_shape().as_list())
    print("Pass line 234")
    all_vars = tf.trainable_variables()
    print(
        "Total parameters' number: %d"
        % (np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))
    )
