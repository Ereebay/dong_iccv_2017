import os
import errno
import torch

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def label2matrix(label):
    labellist = [0] * 312
    for i in label:
        index = int(i)
        labellist[index-1] = 1

    return labellist
