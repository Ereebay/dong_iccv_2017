import tensorflow as tf



def vggclassifier(input):
    net_0 = input
    net_1 = tf.layers.dense(inputs=net_0, units=4096, activation=tf.nn.relu)
    net_1 = tf.layers.dropout(inputs=net_1)

    net_2 = tf.layers.dense(inputs=net_1, units=4096, activation=tf.nn.relu)
    net_2 = tf.layers.dropout(inputs=net_2)

    output = tf.layers.dense(inputs=net_2, units=1000)

    return output

def imgencoder(input):
    net_input = input

    net_conv1 = tf.layers.conv2d(inputs=net_input,
                                 filters=128,
                                 kernel_size=3,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 use_bias=False)

    net_conv2 = tf.layers.conv2d(inputs=net_conv1,
                                 filters=256,
                                 kernel_size=4,
                                 strides=2,
                                 padding='same',
                                 use_bias=False)
    net_conv2 = tf.layers.batch_normalization(net_conv2)
    net_conv2 = tf.nn.relu(net_conv2)

    net_conv3 = tf.layers.conv2d(inputs=net_conv2,
                                 filters=512,
                                 kernel_size=4,
                                 strides=2,
                                 padding='same',
                                 use_bias=False)
    net_conv3 = tf.layers.batch_normalization(net_conv3)
    net_conv3 = tf.nn.relu(net_conv3)

    output = net_conv3

    return output

def imgdecoder(input):
    net_input = input

    net_conv1 =
    net_conv1 = tf.layers.conv2d(inputs=net_conv1,
                                 filters=256,
                                 kernel_size=3,
                                 padding='same',
                                 use_bias=False)
    net_conv1 = tf.layers.batch_normalization(net_conv1)
    net_conv1 = tf.nn.relu(net_conv1)

    net_conv2 =
    net_conv2 = tf.layers.conv2d(inputs=net_conv2,
                                 filters=128,
                                 kernel_size=3,
                                 padding='same',
                                 use_bias=False)
    net_conv2 = tf.layers.batch_normalization(net_conv2)
    net_conv2 = tf.nn.relu(net_conv2)

    net_conv3 = tf.layers.conv2d(inputs=net_conv2,
                                 filters=3,
                                 kernel_size=3,
                                 padding='same',
                                 activation=tf.nn.tanh)
    output = net_conv3

    self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(512, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 128, 3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 3, 3, padding=1),
        nn.Tanh()
    )

def visualsemanticembedding(input, img, txt, embed_ndim):

