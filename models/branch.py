import torch
import torch.nn as nn


class DCRBranch(nn.Module):

    '''Branch Network for DCR'''

    def __init__(self, num_classes, in_channels, mid_channels, normalized_embeddings=False):
        super().__init__()
        self.normalized_embeddings = normalized_embeddings
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, num_classes)
        self.relu = nn.PReLU()

    def forward(self, img):
        if self.normalized_embeddings:
            representation = torch.nn.functional.normalize(self.fc1(img))
        else:
            representation = self.fc1(img)
        classification = self.fc2(self.relu(representation))
        return representation, classification
