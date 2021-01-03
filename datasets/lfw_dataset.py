import os

from .base_dataset import BaseDataset


class LFWDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 subset=0,
                 mode='test', 
                 image_transform=None,
                 scale_size=[8, 12, 16, 20],
                 task='R'):
        super(LFWDataset, self).__init__(
            data_root, 'LFW', subset, mode, image_transform,
            scale_size, task, False)

    def _scan(self):
        object_dir = os.path.join(self.data_root, 'LFW_120x120_120')
        self.file_list = []
        if self.task in ['R', 'IR']:
            for idx, object_name in enumerate(os.listdir(object_dir)):
                object_path = os.path.join(object_dir, object_name)
                subject_id = int(idx)
                if self._id_check(int(subject_id)):
                    self.file_list.extend([(os.path.join(object_path, x), subject_id) for x in os.listdir(object_path)])
        elif self.task in ['V', 'IV']:
            with open(os.path.join(self.data_root, 'lfw_test_pair.txt'), 'r') as f:
                lines = f.readlines()
            # split lines
            lines = list(map(lambda x:x.split(), lines))
            # format fields (format fullpath and cast labels from str to int)
            lines = list(map(
                lambda x:(os.path.join(object_dir, x[0]), os.path.join(object_dir, x[1]), int(x[2])),
                lines))
            # filter images that do not exist
            lines = list(filter(lambda x: os.path.isfile(x[0]) and os.path.isfile(x[1]), lines))
            lines = list(map(lambda x: ('+'.join(x[0:-1]), x[-1]), lines))
            self.file_list = lines
