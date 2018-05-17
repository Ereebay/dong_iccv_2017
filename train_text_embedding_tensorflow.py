
import os
import tensorlayer as tl
import tensorflow as tf
from utils import *
from tensorlayer.cost import *
import pickle
import numpy as np
import nltk
import time

from model_tensorflow import *


batch_size = 64
image_size = 64


print("Loading data from pickle ...")
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)
images_train = np.array(images_train)
images_test = np.array(images_test)


ni = int(np.ceil(np.sqrt(batch_size)))
exists_or_mkdir("pretrain_encoder")
exists_or_mkdir("checkpoint")
save_dir = "checkpoint"


def main_train():
    ##Define model
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_relevant_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], nmae='relevant_caption_input')

    ## training inference for text-to-image mapping
    net_cnn = cnn_encoder(t_real_image, reuse=False)
    x = net_cnn
    v = rnn_embed(t_real_caption, reuse=False)
    x_w = cnn_encoder(t_wrong_image, reuse=True)
    v_w = rnn_embed(t_wrong_caption, reuse=True)

    alpha = 0.2 # margin alpha
    rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
                tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

    #inference
    net_rnn = rnn_embed(t_real_caption, reuse=True)
    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.0002
    lr_decay = 0.5  # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100  # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    cnn_vars = tf.trainable_variables()
    rnn_vars = tf.trainable_variables()

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
    optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)  # optimizer = tf.train.GradientDescentOptimizer(lre)
    rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    # load the latest checkpoints
    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
    net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')

    load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
    load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
    n_epoch = 300
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)
    # exit()
    for epoch in range(0, n_epoch + 1):
        start_time = time.time()

        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            ## get matched text
            idexs = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)
            b_real_caption = captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
            ## get real image
            b_real_images = images_train[
                np.floor(np.asarray(idexs).astype('float') / n_captions_per_image).astype('int')]
            # save_images(b_real_images, [ni, ni], 'samples/step1_gan-cls/train_00.png')
            ## get wrong caption
            idexs = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)
            b_wrong_caption = captions_ids_train[idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
            ## get wrong image
            idexs2 = get_random_int(min=0, max=n_images_train - 1, number=batch_size)
            b_wrong_images = images_train[idexs2]

            b_real_images = threading_data(b_real_images, prepro_img,
                                           mode='train')  # [0, 255] --> [-1, 1] + augmentation
            b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')
            ## updates text-to-image mapping
            if epoch < 50:
                errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
                    t_real_image: b_real_images,
                    t_wrong_image: b_wrong_images,
                    t_real_caption: b_real_caption,
                    t_wrong_caption: b_wrong_caption})
            else:
                errRNN = 0


            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, rnn_loss: %.8f" \
                  % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errRNN))

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time() - start_time))

        ## save model
        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name, sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name + str(epoch), sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name + str(epoch), sess=sess)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                        help='train, train_encoder, translation')

    args = parser.parse_args()

    if args.mode == "train":
        main_train()

    ## you would not use this part, unless you want to try style transfer on GAN-CLS paper
    # elif args.mode == "train_encoder":
    #     main_train_encoder()
    #
    # elif args.mode == "translation":
    #     main_transaltion()

#

