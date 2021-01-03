import torch

from .casia_dataset import CasiaDataset


class DebugDataset(CasiaDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_image = torch.rand(3, 120, 120)
        self._debug_label = torch.tensor(1)

    def __getitem__(self, idx):
        return {
            "data": {str(i):self._debug_image for i in range(len(self.scale_size)+1)},
            "label": self._debug_label,
            'image_path': "fuck"
        }

    def __len__(self):
        return 200