import os, sys
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import AccuracyCalculator, master_only, KFold, eval_acc, find_best_threshold


class BaseDataset(Dataset):

    '''Dataset base class:
    BaseDataset should provide basic methods for iteration (__getitem__) and
    evaluation (both retrieval and verification).

    Args:
        task (str): 'R': retrieval; 'V': verification;
                    'IR': intra-scale retrieval; 'IV': intra-scale verification

    Notes:
    `__getitem__` should format output in a dict:
        {
            'data': 
            {
                '0': tensor or ndarray of images from highest resolution, shape:[C,H0,W0]
                '1': tensor or ndarray of images from other resolution, shape:[C,H1,W1]
                ...
            }
            'label': scalar indicating the label
            'image_path': absolute path of the images
        }

    `evaluate` should format output in a dict:
        {
            'criterion': value of a certain metric for early stopping check
            '0': a dict containing metrics for scale 0
            {
                'metric1': scalar,
                'metric2': scalar,
                ...
            }
            '1': a dict containing metrics for scale 1
            {
                'metric1': scalar,
                'metric2': scalar,
                ...
            }
            ...
        }
    '''

    def __init__(self,
                 data_root,
                 sub_dir,
                 subset=0,
                 mode='train',
                 image_transform=None,
                 scale_size=[],
                 task='R',
                 load_into_memory=False):
        super(BaseDataset, self).__init__()

        self.data_root = os.path.join(data_root, sub_dir)
        self.image_transform = image_transform
        self.scale_size = scale_size
        self.subset = subset
        if mode == 'test': mode = 'val'
        self.mode = mode
        assert task in ['R', 'V', 'IV', 'IR']
        self.task = task[-1]
        self.intra_mode = task.startswith('I')
        self.load_into_memory = load_into_memory

        if self.image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize([120, 120]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        self._scan()
        self._num_classes = len(np.unique(list(map(lambda x:x[1], self.file_list))))
        self._num_scales = len(scale_size) + 1

    def _id_check(self, subject_id):
        return True

    def _scan(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.load_into_memory:
            return self.file_list[idx]

        path, label = self.file_list[idx]
        if self.task == 'R':
            ref_image = Image.open(path).convert('RGB')
            qry_image = ref_image.copy()
            data_dict = {'0': self.image_transform(ref_image)}
        elif self.task == 'V':
            path = path.split('+')
            ref_image = Image.open(path[0]).convert('RGB')
            qry_image = Image.open(path[1]).convert('RGB')
            data_dict = {'0': self.image_transform(ref_image)}

        if self.intra_mode:
            data_dict['ref_0'] = data_dict['0']
            data_dict['0'] = self.image_transform(qry_image)

        for index, scale_size in enumerate(self.scale_size):
            resized_img = qry_image.resize((scale_size, scale_size), Image.ANTIALIAS)
            data_dict['{}'.format(index + 1)] = self.image_transform(resized_img)
            if self.intra_mode:
                resized_img = ref_image.resize((scale_size, scale_size), Image.ANTIALIAS)
                data_dict['ref_{}'.format(index + 1)] = self.image_transform(resized_img)

        return {
            "data": data_dict,
            "label": label,
            'image_path': self.file_list[idx][0]}

    @master_only
    def evaluate(self, embeddings, labels):
        if self.intra_mode:
            ref_embeddings = {k.split('_')[-1]:v for k, v in embeddings.items() if k.startswith('ref_')}
            embeddings = dict(filter(lambda d: not d[0].startswith('ref_'), embeddings.items()))
        else:
            ref_embeddings = embeddings.pop('0')
            ref_embeddings = {k:ref_embeddings for k in embeddings.keys()}
        qry_embeddings = embeddings
        if self.task == 'R':
            return self.evaluate_retrieval(ref_embeddings, qry_embeddings, labels)
        elif self.task == 'V':
            return self.evaluate_verification(ref_embeddings, qry_embeddings, labels)

    def evaluate_verification(self, ref_embeddings, qry_embeddings, labels):
        metric_dict = {}
        for k in qry_embeddings.keys():
            cos_sim = torch.nn.functional.cosine_similarity(ref_embeddings[k], qry_embeddings[k])
            predictions = torch.stack((cos_sim, labels), dim=1).numpy()

            accs, thrs = [], []
            thresholds = np.arange(-1.0, 1.0, 0.005)
            for train_idx, test_idx in KFold(n=len(self), n_folds=10):
                best_thr = find_best_threshold(thresholds, predictions[train_idx])
                accs.append(eval_acc(best_thr, predictions[test_idx]))
                thrs.append(best_thr)
            metric_dict[k] = dict(
                acc=np.mean(accs),
                std=np.std(accs),
                thr=np.mean(thrs)
            )
        metric_dict['criterion'] = np.mean(np.array([metric_dict[k]['acc'] for k in metric_dict.keys()]))
        return metric_dict

    def evaluate_retrieval(self, ref_embeddings, qry_embeddings, labels):
        # assert consistent numbers of queries and references
        assert all(len(ref_embeddings[k]) == len(qry_embeddings[k]) for k in qry_embeddings.keys())

        acc_calculator = AccuracyCalculator()
        metric_dict = {}
        for k in qry_embeddings.keys():
            metric_dict[k] = acc_calculator.get_accuracy(qry_embeddings[k].numpy(
            ), ref_embeddings[k].numpy(), labels.numpy(), labels.numpy(), True)
        ## criterion as avg precision over scales
        metric_dict['criterion'] = np.mean(np.array([metric_dict[k]['precision_at_1'] for k in metric_dict.keys()]))
        return metric_dict

    def __len__(self):
        return len(self.file_list)

    @property
    def label_list(self):
        return list(map(lambda x:x[1], self.file_list))

    @property
    def num_scales(self):
        return self._num_scales

    @property
    def num_classes(self):
        return self._num_classes
