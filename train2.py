import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pyfasttext import FastText

from model import VisualSemanticEmbedding
from model import Generator, Discriminator
from model import *
from data import ReedICML2016
from data import TextDataset
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=False,
                    default='/home/eree/Documents/StackGAN-v2/data/birds')
parser.add_argument('--img_root', type=str, required=False,
                    default='/home/eree/Documents/StackGAN-v2/data/CUB_200_2011/CUB_200_2011/images',
                    help='root directory that contains images')
parser.add_argument('--caption_root', type=str, required=False,
                    default='/home/eree/Documents/StackGAN-v2/data/CUB_200_2011/cub_icml',
                    help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, required=False, default='trainvalclasses.txt',
                    help='text file that contains training classes')
parser.add_argument('--fasttext_model', type=str, required=False,
                    default='/home/eree/Documents/StackGAN-v2/data/CUB_200_2011/wiki.en.bin',
                    help='pretrained fastText model (binary file)')
parser.add_argument('--text_embedding_model', type=str, required=False, default='./models/text_embedding_birds.pth',
                    help='pretrained text embedding model')
parser.add_argument('--save_filename', type=str, required=False, default='./models/birds_vgg_new.pth',
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--use_vgg', action='store_true', default=True,
                    help='use pretrained VGG network for image encoder')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True


def preprocess(img, desc, len_desc, txt_encoder):
    img = img.cuda() if not args.no_cuda else img
    desc = desc.cuda() if not args.no_cuda else desc

    len_desc = len_desc.numpy()
    sorted_indices = np.argsort(len_desc)[::-1]
    new_sorted_indices = sorted_indices.copy()
    original_indices = np.argsort(sorted_indices)
    packed_desc = nn.utils.rnn.pack_padded_sequence(
        desc[new_sorted_indices, ...].transpose(0, 1),
        len_desc[sorted_indices]
    )
    _, txt_feat = txt_encoder(packed_desc)
    txt_feat = txt_feat.squeeze()
    txt_feat = txt_feat[original_indices, ...]

    txt_feat_np = txt_feat.data.cpu().numpy() if not args.no_cuda else txt_feat.data.numpy()
    txt_feat_mismatch = torch.Tensor(np.roll(txt_feat_np, 1, axis=0))
    txt_feat_mismatch = Variable(txt_feat_mismatch.cuda() if not args.no_cuda else txt_feat_mismatch)
    txt_feat_np_split = np.split(txt_feat_np, [txt_feat_np.shape[0] // 2])
    txt_feat_relevant = torch.Tensor(np.concatenate([
        np.roll(txt_feat_np_split[0], -1, axis=0),
        txt_feat_np_split[1]
    ]))
    txt_feat_relevant = Variable(txt_feat_relevant.cuda() if not args.no_cuda else txt_feat_relevant)
    return img, txt_feat, txt_feat_mismatch, txt_feat_relevant


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
             (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if m.weight.requires_grad:
            nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        if m.weight.requires_grad:
            m.weight.data.normal_(1.0, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        if m.weight.requires_grad:
            nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0.0)


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    # word_embedding = FastText(args.fasttext_model)

    print('Loading a dataset...')
    # train_data = ReedICML2016(args.img_root,
    #                           args.caption_root,
    #                           args.trainclasses_file,
    #                           word_embedding,
    #                           args.max_nwords,
    #                           transforms.Compose([
    #                               transforms.Scale(74),
    #                               transforms.RandomCrop(64),
    #                               transforms.RandomHorizontalFlip(),
    #                               transforms.ToTensor()
    #                           ]))
    image_transform = transforms.Compose([
        transforms.Resize(int(256 * 76 / 64)),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip()])
    train_data = TextDataset(args.data_dir, 'train',
                             base_size=64,
                             transform=image_transform)
    print('num_samples: %d' % train_data.__len__())
    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   drop_last=True,)

    # word_embedding = None

    # pretrained text embedding model
    # print('Loading a pretrained text embedding model...')
    # txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    # txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    # txt_encoder = txt_encoder.txt_encoder
    # for param in txt_encoder.parameters():
    #     param.requires_grad = False

    # G = Generator(use_vgg=args.use_vgg)
    # D = Discriminator()
    # Load network
    G = NewG_Net()
    G.apply(weights_init)
    G = torch.nn.DataParallel(G, device_ids=[0])
    G.cuda()
    D = []
    D.append(NEWD_NET64())
    D.append(D_NET128())
    D.append(D_NET256())
    for i in range(len(D)):
        D[i].apply(weights_init)
        D[i] = torch.nn.DataParallel(D[i], device_ids=[0])
        D[i].cuda()
    # AllPARA=[]
    # G_para = G.named_parameters()
    # for name, prar in G_para:
    #     if prar.requires_grad==False:
    #         AllPARA.append(name)
    # for i in range(len(D)):
    #     D_para = D[i].named_parameters()
    #     for name, para in D_para:
    #         if para.requires_grad==False:
    #             AllPARA.append(name)
    #
    # print(AllPARA)
    # inception_model = INCEPTION_V3().cuda()
    # inception_model.eval()

    # if not args.no_cuda:
    #     txt_encoder.cuda()
    #     G.cuda()
    #     D.cuda()

    g_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, G.parameters()),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    d_optimizers = []
    d_lr_scheduler = []
    for i in range(3):
        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, D[i].parameters()),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
        d_optimizers.append(opt)
        scheduler = lr_scheduler.StepLR(opt, 100, args.lr_decay)
        d_lr_scheduler.append(scheduler)
    # d_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, D.parameters()),
    #                                lr=args.learning_rate, betas=(args.momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    # d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, args.lr_decay)

    predictions = []
    count = 0
    mean = 0
    std = 0
    for epoch in range(args.num_epochs):
        start_t = time.time()

        print('start training')
        # training loop
        avg_D_real_loss = 0
        avg_D_real_m_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_kld = 0
        criterion = nn.BCELoss()
        criterion.cuda()
        # for i, (img, desc, len_desc, classid) in enumerate(train_loader):
        for index, (img, embedding, wrongembedding, relevant) in enumerate(train_loader):
            temp = img
            img_norm=[]
            for i in range(3):
                img_norm.append(img[i] * 2 - 1)
                img_norm[i].cuda()
            txt_feat = embedding.cuda()
            txt_feat_mismatch = wrongembedding.cuda()
            txt_feat_relevant = relevant.cuda()
            steptime = time.time()

            # img, txt_feat, txt_feat_mismatch, txt_feat_relevant = \
            #     preprocess(img, desc, len_desc, txt_encoder)
            # img_norm = img * 2 - 1
            # test = img_norm
            # after = (test.data + 1) * 0.5
            # vgg = vgg_normalize(img.data)
            vgg = torch.stack([vgg_normalize(image) for image in temp[0].data])
            vgg.cuda()

            # img_norm = vgg
            # vgg_norm = vgg * 2 - 1
            # ONES = torch.ones(24)
            # ZEROS = torch.zeros(24)
            ONES = torch.FloatTensor(16).fill_(1)
            ZEROS = torch.FloatTensor(16).fill_(0)
            if not args.no_cuda:
                ONES, ZEROS = ONES.cuda(), ZEROS.cuda()
            fake, z_mean, z_log_stddev = G(vgg, txt_feat_relevant)
            for i in range(3):
                # UPDATE DISCRIMINATOR
                D[i].zero_grad()
                # real image with matching text
                real_logit = D[i](img_norm[i], txt_feat)
                # real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
                real_loss = criterion(real_logit, ONES)
                avg_D_real_loss += real_loss.item()
                # real_loss.backward()
                # real image with mismatching text
                real_m_logit = D[i](img_norm[i], txt_feat_mismatch)
                # real_m_loss = 0.5 * F.binary_cross_entropy_with_logits(real_m_logit, ZEROS)
                real_m_loss = 0.5 * criterion(real_m_logit, ZEROS)
                avg_D_real_m_loss += real_m_loss.item()
                # real_m_loss.backward()
                # synthesized image with semantically relevant text
                fake_logit = D[i](fake[i].detach(), txt_feat_relevant)
                # fake_loss = 0.5 * F.binary_cross_entropy_with_logits(fake_logit, ZEROS)
                fake_loss = 0.5 * criterion(fake_logit, ZEROS)
                avg_D_fake_loss += fake_loss.item()
                # fake_loss.backward()
                errD = real_loss + real_m_loss + fake_loss
                errD.backward()
                d_optimizers[i].step()


            # UPDATE GENERATOR
            G.zero_grad()
            G_loss=0

            KLD_element = z_mean.pow(2).add_(z_log_stddev.exp()).mul_(-1).add_(1).add_(z_log_stddev)
            kld = torch.mean(KLD_element).mul_(-0.5)
            # kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.item()
            for i in range(3):
                fake_logit = D[i](fake[i], txt_feat_relevant)
                # fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
                fake_loss = criterion(fake_logit, ONES)
                avg_G_fake_loss += fake_loss.item()
                G_loss += fake_loss
            G_loss = G_loss+kld
            G_loss.backward()

            g_optimizer.step()

            # pred = inception_model(fake.detach())
            # predictions.append(pred.data.cpu().numpy())
            #
            #
            # # compute inception score
            # if len(predictions) > 500:
            #     predictions = np.concatenate(predictions, 0)
            #     mean, std = compute_inception_score(predictions, 10)
            #     print('mean:', mean, 'std:', std)
            #     predictions = []

            # print(
            #     'Epoch [%d/%d], Iter [%d/%d], Time [%.4f], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f, IS: %.4f, std: %.4f'
            #     % (epoch + 1, args.num_epochs, index + 1, len(train_loader), time.time() - steptime,
            #        avg_D_real_loss / (index + 1),
            #        avg_D_real_m_loss / (index + 1), avg_D_fake_loss / (index + 1), avg_G_fake_loss / (index + 1),
            #        avg_kld / (index + 1)), mean, std)
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Iter [{index + 1}/{len(train_loader)}],'
                  f'Time [{(time.time() - steptime):.2f}], D_real:{(avg_D_real_loss / (index + 1)):.4f},'
                  f'D_mis:{(avg_D_real_m_loss / (index + 1)):.4f}, D_fake:{(avg_G_fake_loss / (index + 1)):.4f},'
                  f'KLD:{(avg_kld / (index + 1)):.4f}, IS:{mean:.4f}, STD:{std:.4f}')
        # d_lr_scheduler.step()
        for i in range(3):
            d_lr_scheduler[i].step()

        g_lr_scheduler.step()
        # for i in range(64):
        #     save_image(fake.data[2][i], './examples/kdd/birds/epoch_%d_%d.png' %(epoch+1, i))
        # save_image(fake.data, './examples/epochnorm_%d.png' % (epoch + 1), normalize=True)
        save_image((fake[0].data + 1) / 2, './examples/64epoch_%d.png' % (epoch + 1))
        save_image((fake[1].data + 1) / 2, './examples/128epoch_%d.png' % (epoch + 1))
        save_image((fake[2].data + 1) / 2, './examples/256epoch_%d.png' % (epoch + 1))

        torch.save(G.state_dict(), args.save_filename)
