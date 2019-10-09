import argparse
import numpy as np
import time
from PIL import Image
import errno
import json
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import *
from data import TextDataset
import os

G_PATH = '/home/eree/Documents/dong_iccv_2017/models/birds_vgg_new.pth'
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
parser.add_argument('--batch_size', type=int, default=20,
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

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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

def save_singleimages(images, filenames, save_dir, split_dir, sentenceID, imsize, attribute_values, captions):
    all_dict = []

    for i in range(images.size(0)):
        s_tmp = '%s/single_samples/%s/%s' % \
                (save_dir, split_dir, filenames[i])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        caption = captions[sentenceID][i]
        from functools import reduce
        attribute_values = reduce(lambda x, y: torch.cat((x.view(-1, 20),y.view(1,-1)), 0), attribute_values)
        attribute_value = attribute_values.transpose(0,1)
        attribute_value = attribute_value[i].tolist()
        fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
        savepath = '%s_%d_sentence%d.png' % (filenames[i], imsize, sentenceID)
        # range from [-1, 1] to [0, 255]
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)
        datum = {'image_file': savepath, 'attribute_value': attribute_value, 'caption': caption}
        print('file of image:', savepath)
        print('caption:', caption)
        all_dict.append(datum)

    return all_dict
def save_superimages(images_list, filenames,
                     save_dir, split_dir, imsize):
    batch_size = images_list[0].size(0)
    num_sentences = len(images_list)
    for i in range(batch_size):
        s_tmp = '%s/super/%s/%s' % \
                (save_dir, split_dir, filenames[i])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        #
        savename = '%s_%d.png' % (s_tmp, imsize)
        super_img = []
        for j in range(num_sentences):
            img = images_list[j][i]
            # print(img.size())
            img = img.view(1, 3, imsize, imsize)
            # print(img.size())
            super_img.append(img)
            # break
        super_img = torch.cat(super_img, 0)
        vutils.save_image(super_img, savename, nrow=10, normalize=True)
def save_singleimages2(images, filenames, save_dir, split_dir, sentenceID, imsize):

    for i in range(images.size(0)):
        s_tmp = '%s/single_samples/%s/%s' % \
                (save_dir, split_dir, filenames[i])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
        savepath = '%s_%d_sentence%d.png' % (filenames[i], imsize, sentenceID)
        # range from [-1, 1] to [0, 255]images
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)
        print('file of image:', savepath)
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

    print('Loading a dataset...')

    image_transform = transforms.Compose([
        transforms.Resize(int(256 * 76 / 64)),
        transforms.RandomCrop(256),
        #transforms.RandomHorizontalFlip()
        ])
    train_data = TextDataset(args.data_dir, 'test',
                             base_size=64,
                             transform=image_transform)
    print('num_samples: %d' % train_data.__len__())
    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_threads,
                                   drop_last=True,)


    # Load network
    G = NewG_Net()
    G.apply(weights_init)
    G = torch.nn.DataParallel(G, device_ids=[0])
    print(G)

    state_dict = torch.load(G_PATH, map_location=lambda storage, loc: storage)
    G.load_state_dict(state_dict)
    print('Load', G_PATH)

    # the path to save generated images
    save_dir = './outputs'

    nz = 100
    noise = torch.FloatTensor(args.batch_size, nz)
    G.cuda()
    noise = noise.cuda()

    with torch.no_grad():
        G.eval()
        for step, data in enumerate(train_loader, 0):
            imgs, t_embeddings, filenames, captions, attribute_values = data
            t_embeddings = t_embeddings.cuda()

            embedding_dim = t_embeddings.size(1)
            batch_size = imgs[0].size(0)
            noise.data.resize_(batch_size, nz)
            noise.data.normal_(0, 1)
            temp = imgs
            vgg = torch.stack([vgg_normalize(image) for image in temp[0].data])
            vgg.cuda()

            fake_img_list = []
            if not os.path.isdir(save_dir):
                print('Make a new folder: ', save_dir)
                mkdir_p(save_dir)
            f = open('%s/info.json' % save_dir, 'a')
            for i in range(embedding_dim):
                fake_imgs, _, _ = G(vgg, t_embeddings[:, i, :])
                fake_img_list.append(fake_imgs[2].data.cpu())
                all_dict = save_singleimages(fake_imgs[-1], filenames, save_dir, 'test', i, 256, attribute_values, captions)
                save_singleimages2(fake_imgs[-2], filenames,
                                        save_dir, '128', i, 128)
                save_singleimages2(fake_imgs[-3], filenames,
                                        save_dir, '64', i, 64)
                for item in all_dict:
                    json.dump(item, f)
            f.close()
            save_superimages(fake_img_list, filenames,
                                save_dir, 'test', 256)