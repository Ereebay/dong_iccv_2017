import tensorflow as tf
from tensorlayer.layers import *



batch_size = 64
image_size = 64     # 64 x 64
c_dim = 3           # for rgb



def vggclassifier(input):
    net_0 = input
    net_1 = tf.layers.dense(inputs=net_0, units=4096, activation=tf.nn.relu)
    net_1 = tf.layers.dropout(inputs=net_1)

    net_2 = tf.layers.dense(inputs=net_1, units=4096, activation=tf.nn.relu)
    net_2 = tf.layers.dropout(inputs=net_2)

    output = tf.layers.dense(inputs=net_2, units=1000)

    return output

def residualblcok(input,w_init,gamma_init):
    net_input = input
    net_h0 = tf.layers.conv2d(inputs=net_input,
                              filters=512,
                              kernel_size=3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=w_init)
    net_h0 = tf.layers.batch_normalization(net_h0,gamma_initializer=gamma_init)
    net_h0 = tf.nn.relu(net_h0)

    net_h1 = tf.layers.conv2d(inputs=net_h0,
                              filters=512,
                              kernel_size=3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=w_init)
    net_h1 = tf.layers.batch_normalization(net_h1,gamma_initializer=gamma_init)
    return net_h1

def generator_simple(input_img, input_txt=None, reuse=False):

    s = image_size
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    gf_dim = 128

    with tf.variable_scope("generator", reuse=reuse):
        ## imgencoder
        net_input = input_img

        net_h0 = tf.layers.conv2d(inputs=net_input,
                                  filters=128,
                                  kernel_size=3,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_initializer=w_init)

        net_h1 = tf.layers.conv2d(inputs=net_h0,
                                  filters=256,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h1 = tf.layers.batch_normalization(net_h1,gamma_initializer=gamma_init)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d(inputs=net_h1,
                                  filters=512,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h2 = tf.layers.batch_normalization(net_h2,gamma_initializer=gamma_init)
        net_h2 = tf.nn.relu(net_h2)

        net_output = net_h2
        img_feat = net_output

        txt_feat = input_txt
        txt_feat =tf.layers.dense(txt_feat,128,activation=tf.nn.leaky_relu,kernel_initializer=w_init)
        txt_feat = tf.expand_dims(tf.expand_dims(txt_feat,1),1)
        txt_feat = tf.tile(txt_feat,multiples=[1,16,16,1])
        fusion = tf.concat([img_feat,txt_feat],3)
        fusion = tf.layers.conv2d(inputs=fusion,
                                  filters=512,
                                  kernel_size=3,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        fusion = tf.layers.batch_normalization(fusion, gamma_initializer=gamma_init)

        fusion = residualblcok(fusion,w_init,gamma_init)
        fusion = residualblcok(fusion,w_init,gamma_init)
        fusion = residualblcok(fusion, w_init, gamma_init)
        fusion = residualblcok(fusion, w_init, gamma_init)

        ## imgdecoder
        net_input = fusion

        net_h0 = tf.layers.conv2d_transpose(inputs=net_input,
                                            filters=256,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=w_init)
        net_h0 = tf.layers.batch_normalization(net_h0,gamma_initializer=gamma_init)
        net_h0 = tf.nn.relu(net_h0)

        net_h1 = tf.layers.conv2d_transpose(inputs=net_h0,
                                            filters=128,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=w_init)
        net_h1 = tf.layers.batch_normalization(net_h1,gamma_initializer=w_init)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d_transpose(inputs=net_h1,
                                            filters=3,
                                            kernel_size=3,
                                            padding='same',
                                            kernel_initializer=w_init)
        logits = net_h2
        net_h2 = tf.nn.tanh(net_h2)
        output = net_h2

        return output,logits

def discriminator_simple(input_image,input_txt=None,reuse = False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64



    with tf.variable_scope("discriminator", reuse=reuse):
        net_input = input_image

        net_h0 = tf.layers.conv2d(inputs=net_input,
                                  filters=64,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=w_init
                                  )
        net_h1 = tf.layers.conv2d(inputs=net_h0,
                                  filters=128,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h1 = tf.layers.batch_normalization(net_h1,gamma_initializer=gamma_init)
        net_h1 = tf.nn.leaky_relu(net_h1)

        net_h2 = tf.layers.conv2d(inputs=net_h1,
                                  filters=256,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init)
        net_h2 = tf.nn.leaky_relu(net_h2)

        net_h3 = tf.layers.conv2d(inputs=net_h2,
                                  filters=512,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h3 = tf.layers.batch_normalization(net_h3,gamma_initializer=gamma_init)

        txt_feat = input_txt
        txt_feat = tf.layers.dense(txt_feat,128,activation=tf.nn.leaky_relu,kernel_initializer=w_init)
        txt_feat = tf.expand_dims(tf.expand_dims(txt_feat,1),1)
        txt_feat = tf.tile(txt_feat,multiples=[1,4,4,1])

        net_h3 = tf.concat([net_h3,txt_feat],3)
        net_h3 = tf.layers.conv2d(inputs=net_h3,
                                  filters=512,
                                  kernel_size=1,
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init)
        net_h3 = tf.nn.leaky_relu(net_h3)

        net_h4 = tf.layers.conv2d(inputs=net_h3,
                                  filters=1,
                                  kernel_size=4,
                                  kernel_initializer=w_init)
        logits = net_h4
        output = tf.nn.sigmoid(logits)

    return output,logits


t_dim = 128  # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 300
keep_prob = 1.0


def cnn_encoder(inputs, reuse=False, name='cnnftxt'):
    """ 64x64 --> t_dim, for text-image mapping """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope(name, reuse=reuse):
        net_input = inputs
        net_h0 = tf.layers.conv2d(inputs=net_input,
                                  filters=128,
                                  kernel_size=3,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_initializer=w_init)

        net_h1 = tf.layers.conv2d(inputs=net_h0,
                                  filters=256,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h1 = tf.layers.batch_normalization(net_h1,
                                               gamma_initializer=gamma_init)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d(inputs=net_h1,
                                  filters=512,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=w_init)
        net_h2 = tf.layers.batch_normalization(net_h2,
                                               gamma_initializer=gamma_init)
        net_h2 = tf.nn.relu(net_h2)

        net_h3 = tf.layers.flatten(net_h2)
        net_h3 = tf.layers.dense(inputs=net_h3, units=128, kernel_initializer=w_init)

        net_output = net_h3

        return net_output


def rnn_embed(input_seqs, reuse=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob,keep_prob),
                     initializer = w_init,
                     sequence_length = retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')
        return network.outputs
