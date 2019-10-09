import os
import numpy as np
from PIL import Image
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import random
import torch
import torch.utils.data as data
import torchfile
import torchvision.transforms as transforms
from miscc.config import cfg

def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())

def get_imgs(img_path, imsize=[64,128], bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(3):
        if i < (2):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    # ret = normalize(transforms.Resize(64)(img))

    return ret

class ReedICML2016(data.Dataset):
    def __init__(self, img_root, caption_root, classes_fllename,
                 word_embedding, max_word_length, img_transform=None):
        super(ReedICML2016, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        self.max_word_length = max_word_length
        self.img_transform = img_transform

        if self.img_transform == None:
            self.img_transform = transforms.ToTensor()

        self.data = self._load_dataset(img_root, caption_root, classes_fllename, word_embedding)


    def _load_dataset(self, img_root, caption_root, classes_filename, word_embedding):
        output = []
        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = torchfile.load(os.path.join(caption_root, cls, filename))
                    # clstest = cls.split('_')
                    # classid = int(clstest[1])
                    raw_desc = datum[b'char']
                    imgdir = str(datum[b'img'], encoding='utf-8')
                    desc, len_desc = self._get_word_vectors(raw_desc, word_embedding)
                    output.append({
                        'img': os.path.join(img_root, imgdir),
                        'desc': desc,
                        'len_desc': len_desc,
                        # 'classid': classid
                    })
        return output

    def _get_word_vectors(self, desc, word_embedding):
        output = []
        len_desc = []
        for i in range(desc.shape[1]):
            words = self._nums2chars(desc[:, i])
            words = split_sentence_into_words(words)
            word_vecs = torch.Tensor([word_embedding[w] for w in words])
            # zero padding
            if len(words) < self.max_word_length:
                word_vecs = torch.cat((
                    word_vecs,
                    torch.zeros(self.max_word_length - len(words), word_vecs.size(1))
                ))
            output.append(word_vecs)
            len_desc.append(len(words))
        return torch.stack(output), len_desc

    def _nums2chars(self, nums):
        chars = ''
        for num in nums:
            chars += self.alphabet[num - 1]
        return chars


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        img = vgg_normalize(img)
        desc = datum['desc']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]
        #classid = datum['classid']
        return img, desc, len_desc#, classid





class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor()])#,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = [64,128]
        # for i in range(cfg.TREE.BRANCH_NUM):
        #     self.imsize.append(base_size)
        #     base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()
        # self.all_info = self.load_all_info(split_dir)

        if split=='train':
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def load_all_info(self, data_dir):
        filepath = os.path.join(data_dir, 'all_info.pickle')
        with open(filepath, 'rb') as f:
            all_info = pickle.load(f)
        print('Load all info from : %s (%d)' % (filepath, len(all_info)))

        return all_info

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        attribute_value = self.all_info[index]['attribute_value']
        # attribute_value = label2matrix(attribute_value)
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        relevant_ix = random.randint(0, len(self.filenames) - 1)
        while(self.class_id[index] != self.class_id[relevant_ix]):
            relevant_ix = random.randint(0, len(self.filenames) - 1)
        relevant = relevant_ix
        if(wrong_ix == index):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_embedings = self.embeddings[wrong_ix, :, :]
        relevant_embeddings = self.embeddings[relevant, :, :]
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        wrong_embedding = wrong_embedings[embedding_ix, :]
        relevant_embedding = relevant_embeddings[embedding_ix, :]

        embedding = embeddings[embedding_ix, :]
        captions = self.captions[self.filenames[index]][embedding_ix]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, embedding, wrong_embedding, relevant_embedding#, key, captions, attribute_value  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # bbox = None
        # data_dir = self.data_dir

        number = str(random.randint(0, 9))
        wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrongkey = self.filenames[wrong_ix]
        captions = self.captions[wrongkey]
        embeddings = self.embeddings[wrong_ix, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # attribute_value = self.all_info[index]['attribute_value']

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key, captions# , attribute_value  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
