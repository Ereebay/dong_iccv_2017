#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk

from utils import *
from model_tensorflow import *
import pickle

batch_size = 64
image_size = 64
###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)
images_train = np.array(images_train)
images_test = np.array(images_test)

# print(n_captions_train, n_captions_test)
# exit()

ni = int(np.ceil(np.sqrt(batch_size)))
tl.files.exists_or_mkdir("samples")
tl.files.exists_or_mkdir("checkpoint")
save_dir = "checkpoint"


def main_train():
    ###======================== DEFIINE MODEL ===================================###
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_relevant_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='relevant_caption_input')

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
    ## training inference for txt2img
    generator = generator_simple
    discriminator = discriminator_simple

    net_fake_image, _ = generator(t_real_image,
                    rnn_embed(t_relevant_caption, reuse=True),
                    reuse=False)

    net_d, disc_fake_image_logits = discriminator(
                    net_fake_image, rnn_embed(t_relevant_caption,reuse=True), reuse=False)
    _, disc_real_image_logits = discriminator(
                    t_real_image, rnn_embed(t_real_caption,reuse=True), reuse=True)
    _, disc_mismatch_logits = discriminator(
                    t_real_image,
                    rnn_embed(t_wrong_caption, reuse=True),
                    reuse=True)

    ## testing inference for txt2img
    net_g, _ = generator(t_real_image,
                    rnn_embed(t_real_caption, reuse=True),
                    reuse=True)

    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
    d_loss = d_loss1 + d_loss2 + d_loss3
    g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')

    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.0002
    lr_decay = 0.5      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    cnn_vars = tf.trainable_variables()
    rnn_vars = tf.trainable_variables()
    d_vars = tf.trainable_variables()
    g_vars = tf.trainable_variables()

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )
    grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
    # e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(e_loss, var_list=e_vars + c_vars)
    optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)# optimizer = tf.train.GradientDescentOptimizer(lre)
    rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    # adam_vars = tl.layers.get_variables_with_name('Adam', False, True)

    ###============================ TRAINING ====================================###
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())


    # saver = tf.train.import_meta_graph('./checkpoint/embed/model.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/embed'))
    # saver = tf.train.import_meta_graph('./checkpoint/gan/model.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('./checkpoint/gan'))

    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
    load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)

    ## seed for generation, z and sentence ids
    sample_size = batch_size
        # sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)
    sample_sentence = ["The petals are white and the stamens are light yellow."] * int(sample_size/ni) + \
                      ["The red flower has no visible stamens."] * int(sample_size/ni) + \
                      ["The petals of the flower have yellow and red stipes."] * int(sample_size/ni) + \
                      ["The petals of the flower have mixed colors of bright yellow and light green."] * int(sample_size/ni) + \
                      ["This light purple flower has a large number of small petals."] * int(sample_size/ni) + \
                      ["This flower has petals of pink and white color with yellow stamens."] * int(sample_size/ni) + \
                      ["The flower shown has reddish petals with yellow edges"] * int(sample_size/ni) +\
                      ["These white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)

    # sample_sentence = captions_ids_test[0:sample_size]
    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID
        # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
        # print(sample_sentence[i])
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    ### get image test
    tmp = get_random_int(min=0,max=n_captions_test-1,number=64)
    idex4 = get_random_int(min=0, max=n_images_test - 1, number=8)
    b_test_image = images_test[idex4]
    imagetest = images_test[np.floor(np.asarray(tmp).astype('float') / n_captions_per_image).astype('int')]
    save_images(b_test_image, [1, 8], 'samples/ori.png')
    for i in [0,8,16,24,32,40,48,56]:
        imagetest[i] = b_test_image[0]
        imagetest[i + 1] = b_test_image[1]
        imagetest[i + 2] = b_test_image[2]
        imagetest[i + 3] = b_test_image[3]
        imagetest[i + 4] = b_test_image[4]
        imagetest[i + 5] = b_test_image[5]
        imagetest[i + 6] = b_test_image[6]
        imagetest[i + 7] = b_test_image[7]
    save_images(imagetest,[8,8],'samples/ori2.png')

    n_epoch = 300
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)
    # exit()
    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
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
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_real_caption = captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
            ## get real image
            b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
            save_images(b_real_images, [ni, ni], 'samples/before/train_00.png')
            ## get wrong caption
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_wrong_caption = captions_ids_train[idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
            ## get wrong image
            idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
            b_wrong_images = images_train[idexs2]

            ## get_relevant caption
            idexs3 = get_random_int(min=0, max = n_captions_train-1, number=batch_size)
            b_rel_caption = captions_ids_train[idexs3]
            b_rel_caption = tl.prepro.pad_sequences(b_rel_caption, padding='post')

            b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
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

            ## updates D
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image: b_real_images,
                            t_wrong_caption: b_wrong_caption,
                            t_real_caption: b_real_caption,
                            t_relevant_caption: b_rel_caption})
            ## updates G
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_image: b_real_images,
                            t_relevant_caption: b_rel_caption})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f,rnn_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG,errRNN))

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([net_g, rnn_embed(t_real_caption,reuse=True)], feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_real_image : imagetest})

            # img_gen = threading_data(img_gen, prepro_img, mode='rescale')
            save_images(img_gen, [ni, ni], 'samples/after/train_{:02d}.png'.format(epoch))

        ## save model
        if (epoch != 0) and (epoch % 10) == 0:
            #tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
            #tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
            save_path = saver.save(sess,"checkpoint/gan/model")
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            #tl.files.save_npz(net_g.all_params, name=net_g_name+str(epoch), sess=sess)
            #tl.files.save_npz(net_d.all_params, name=net_d_name+str(epoch), sess=sess)
            save_path = saver.save(sess,"checkpoint/gan/model")

        # if (epoch != 0) and (epoch % 200) == 0:
        #     sess.run(tf.initialize_variables(adam_vars))
        #     print("Re-initialize Adam")



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
