import torch
from torch.utils.data.sampler import Sampler
import os, sys
sys.path.append('../')
from utils import common_functions as c_f

# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned

    Notes:
        1. The implementation tips:
            a. Create a dictionary mapping each label to corresponding indexs
            b. For each label, randomly take fixed number of index from the map

    """
    def __init__(self, labels, m, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class*len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0]*self.list_size
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.labels)
            for label in self.labels:
                t = self.labels_to_indices[label]
                idx_list[i:i+self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)

if __name__=="__main__":
    labels = [1,1,2,2,3,3,3,4,4,4]
    sampler = MPerClassSampler(labels, 3, 100)
    s_iter = iter(sampler)
    idx_list = []
    for idx in s_iter:
        idx_list.append(idx)
    
    import numpy as np
    idx_list = np.array(idx_list)
    label_list = []
    for idx in idx_list:
        label_list.append(labels[idx])

    print(label_list)