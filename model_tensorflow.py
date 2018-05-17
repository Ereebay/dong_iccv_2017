import tensorflow as tf
from tensorlayer.layers import *

def vggclassifier(input):
    net_0 = input
    net_1 = tf.layers.dense(inputs=net_0, units=4096, activation=tf.nn.relu)
    net_1 = tf.layers.dropout(inputs=net_1)

    net_2 = tf.layers.dense(inputs=net_1, units=4096, activation=tf.nn.relu)
    net_2 = tf.layers.dropout(inputs=net_2)

    output = tf.layers.dense(inputs=net_2, units=1000)

    return output


#
# def imgdecoder(input):
#     net_input = input
#
#     net_conv1 =
#     net_conv1 = tf.layers.conv2d(inputs=net_conv1,
#                                  filters=256,
#                                  kernel_size=3,
#                                  padding='same',
#                                  use_bias=False)
#     net_conv1 = tf.layers.batch_normalization(net_conv1)
#     net_conv1 = tf.nn.relu(net_conv1)
#
#     net_conv2 =
#     net_conv2 = tf.layers.conv2d(inputs=net_conv2,
#                                  filters=128,
#                                  kernel_size=3,
#                                  padding='same',
#                                  use_bias=False)
#     net_conv2 = tf.layers.batch_normalization(net_conv2)
#     net_conv2 = tf.nn.relu(net_conv2)
#
#     net_conv3 = tf.layers.conv2d(inputs=net_conv2,
#                                  filters=3,
#                                  kernel_size=3,
#                                  padding='same',
#                                  activation=tf.nn.tanh)
#     output = net_conv3
#
#     self.decoder = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.Conv2d(512, 256, 3, padding=1, bias=False),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.Conv2d(256, 128, 3, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 3, 3, padding=1),
#         nn.Tanh()
#     )

def imgencoder(input):
    net_input = input

    net_h0 = tf.layers.conv2d(inputs=net_input,
                              filters=128,
                              kernel_size=3,
                              padding='same',
                              activation=tf.nn.relu,
                              use_bias=False)

    net_h1 = tf.layers.conv2d(inputs=net_h0,
                              filters=256,
                              kernel_size=4,
                              strides=2,
                              padding='same',
                              use_bias=False)
    net_h1 = tf.layers.batch_normalization(net_h1)
    net_h1 = tf.nn.relu(net_h1)

    net_h2 = tf.layers.conv2d(inputs=net_h1,
                              filters=512,
                              kernel_size=4,
                              strides=2,
                              padding='same',
                              use_bias=False)
    net_h2 = tf.layers.batch_normalization(net_h2)
    net_h2 = tf.nn.relu(net_h2)

    net_output = net_h2

    return net_output


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
                                  use_bias=False)

        net_h1 = tf.layers.conv2d(inputs=net_h0,
                                  filters=256,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d(inputs=net_h1,
                                  filters=512,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  use_bias=False)
        net_h2 = tf.layers.batch_normalization(net_h2)
        net_h2 = tf.nn.relu(net_h2)

        net_h3 = tf.layers.flatten(net_h2)
        net_h3 = tf.layers.dense(inputs=net_h3, units=128)

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
