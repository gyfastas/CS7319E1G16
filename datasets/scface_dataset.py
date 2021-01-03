import os, sys
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset
from utils import AccuracyCalculator, master_only


class SCfaceDataset(BaseDataset):

    # default setting: using last 50 classes for testing
    total_num_classes = 130
    test_start_index = 50

    def __init__(self,
                 data_root,
                 subset=0,
                 mode='test', 
                 image_transform=None,
                 scale_size=[8, 12, 16, 20],
                 task='R',
                 load_into_memory=False):
        assert task == 'R'
        super(SCfaceDataset, self).__init__(
            data_root, 'SCface', subset, mode, image_transform,
            scale_size, task, load_into_memory)
        self._num_classes = self.total_num_classes
        self._num_scales = 4

    @master_only
    def evaluate(self, embeddings, labels):
        ref_embeddings = embeddings.pop('frontal_0')
        query_embeddings = {}
        scale_idxs = sorted(set([x.split('_')[-1] for x in embeddings.keys()]))
        cam_num = len(set([x.split('_')[0] for x in embeddings.keys()]))
        ref_labels = labels
        query_labels = labels.repeat(cam_num)
        for scale in scale_idxs:
            query_embeddings[scale] = torch.cat([v for (k, v) in embeddings.items() if k.split('_')[-1]==scale])

        acc_calculator = AccuracyCalculator()
        metric_dict = {}
        for k in query_embeddings.keys():
            metric_dict[k] = acc_calculator.get_accuracy(query_embeddings[k].numpy(
            ), ref_embeddings.numpy(), query_labels.numpy(), ref_labels.numpy(), False)
            
        metric_dict['criterion'] = np.array([metric_dict[x]['precision_at_1'] for x in metric_dict.keys()]).mean()
        return metric_dict

    def _scan(self):
        file_list = []
        self.selected_folders =['mugshot_frontal_cropped_all'] + ['surveillance_cameras_distance_{}'.format(x) for x in range(1,4)]
        for folder_name in self.selected_folders:
            folder_path = os.path.join(self.data_root, folder_name)
            
            for root, dirs, files in os.walk(folder_path, topdown=True):
                for name in files:
                    if name.endswith('.jpg') or name.endswith('.JPG'):
                        file_list.append((os.path.join(root, name), int(name.split('_')[0]) - 1))
        self.file_list = self.form_subset(sorted(file_list, key=lambda x:x[1]))
            
    def form_subset(self, file_list):
        '''
        reconstruct image list and label list such that
        multiple scales image return at once.
        
        returns:
            new_image_list: [{'scale_0':, 'scale_1':, 'scale_2':...}, {...}]
            new_label_list: [0,2,1,...]
        '''
        new_image_dict = {}
        new_label_dict = {}

        image_list, label_list = zip(*file_list)
        
        # for surveillance cameras, images are xxx_camxx_x.jpg
        # for mugshot, images are xxx_frontal.JPG
        for image_path, label in zip(image_list, label_list):
            [root, image_name] = image_path.rsplit('/', 1)
            image_id = image_name.split('_')[0]
            cam_id = image_name.split('_')[1].split('.')[0]
            if len(image_name.split('_')) <=2:
                # xxx_frontal.JPG
                dist_id = 0
            else:
                # scale_1 = dist_1
                dist_id = int(image_name.split('_')[-1].split('.')[0])
            if self._id_check(label):
                if self.mode=='train':
                    if cam_id=='frontal':
                        for cam_id in ['cam{}'.format(x) for x in range(1,6)]:
                            name = '{}_{}'.format(image_id, cam_id)
                            if name not in new_image_dict:
                                new_image_dict[name] = dict()
                                new_image_dict[name]['{}'.format(dist_id)] = image_path
                            else:
                                new_image_dict[name]['{}'.format(
                                    dist_id)] = image_path
                            new_label_dict[name] = label
                    else:
                        name = '{}_{}'.format(image_id, cam_id)
                        if name not in new_image_dict:
                            new_image_dict[name] = dict()
                            new_image_dict[name]['{}'.format(dist_id)] = image_path
                        else:
                            new_image_dict[name]['{}'.format(
                                dist_id)] = image_path
                        new_label_dict[name] = label
                else:
                    name = '{}'.format(image_id)
                    if name not in new_image_dict:
                        new_image_dict[name] = dict()
                        new_image_dict[name]['{}_{}'.format(cam_id, dist_id)] = image_path
                    else:
                        new_image_dict[name]['{}_{}'.format(
                            cam_id, dist_id)] = image_path
                    new_label_dict[name] = label
            else:
                continue

        new_image_list = list(new_image_dict.values())
        new_label_list = list(new_label_dict.values())
        return list(zip(new_image_list, new_label_list))
    
    def _id_check(self, label):
        if self.mode in ['test', 'val']:
            return (label >=self.test_start_index)
        elif self.mode=='train':
            return (label <self.test_start_index)
        
    def __getitem__(self, idx):
        image_dict = self.file_list[idx][0]
        image_dict = {k: Image.open(v).convert('RGB') for (k, v) in image_dict.items()}
        for k in image_dict.keys():
            image_dict[k] = self.image_transform(image_dict[k])
        label = self.file_list[idx][1]
        return {'data': image_dict, 'label': label, 'image_path': self.file_list[idx][0]}

