import os
from PIL import Image

import torch

from .base_dataset import BaseDataset

class CasiaDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 subset=0,
                 mode='test', 
                 image_transform=None,
                 scale_size=[8, 12, 16, 20],
                 task='R',
                 load_into_memory=False):
        assert task == 'R'
        self.subject_number = 10575
        super(CasiaDataset, self).__init__(
            data_root, 'Casia', subset, mode, image_transform,
            scale_size, task, load_into_memory)
        self._num_classes = self.subject_number

    def _id_check(self, subject_id):
        v = self.subset / 10
        if subject_id >= v * self.subject_number and subject_id < (v + 0.1) * self.subject_number and self.mode == 'val':
            return True
        elif not (subject_id >= v * self.subject_number and subject_id < (v + 0.1) * self.subject_number) and self.mode == 'train':
            return True
        else:
            return False

    def _scan(self):
        with open(os.path.join(self.data_root, 'casia_cleaned_list.txt'), 'r') as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            path, subject_id = line.strip().split(' ')
            if not self._id_check(int(subject_id)):
                continue
            path = os.path.join(self.data_root, '120x120_120', path)
            if self.load_into_memory:
                image = Image.open(path).convert('RGB')
                data_dict = {'0': self.image_transform(image)}
                for index, scale_size in enumerate(self.scale_size):
                    i = image.resize((scale_size, scale_size), Image.ANTIALIAS)
                    data_dict['{}'.format(index + 1)] = self.image_transform(i)
                self.file_list.append({'data': data_dict, 'label': int(subject_id), 'image_path': path})
            else:
                self.file_list.append((path, int(subject_id)))
