import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn.functional import mse_loss

from .center_buffer import CenterBuffer
from miners import BatchHardMiner


class CMLoss(nn.Module):
    """
    CM Loss
    
    Args:
        num_classes: (int) number of classses (in LRFR, number of persons)
        num_dimensions: (int) dimension of each class center,
                              should be consistent with the dimension of feature embeddings
        num_scales: (int) number of resolutions
        update_factor: (float) momentum factor used for center updating
        center_weight: (float) weight for center loss
        dist_weight: (float) weight for euclidean loss

    Notes:
    `forward` accepts input in the following form:
        embeddings: a dict of feature embeddings from different resolutions
        {
            '0': tensor of embeddings from highest resolution, shape:[N,D]
            '1': tensor of embeddings from other resolution, shape:[N,D]
            ...
        }
        logits: a dict of classification logits from different resolutions
        {
            '0': tensor of logits from highest resolution, shape:[N,C]
            '1': tensor of logits from other resolution, shape:[N,C]
            ...
        }
        labels: tensor of labels, shape:[N]
    """

    def __init__(self,
                 num_classes,
                 num_dimensions,
                 num_scales,
                 update_factor=0.6,
                 center_weight=0.008,
                 dist_weight=0.008,
                 miner=None):
        super(CMLoss, self).__init__()
        self.centers = nn.ModuleDict({
            str(s):CenterBuffer(num_classes, num_dimensions, update_factor) for s in range(num_scales)
        })
        self.center_weight = center_weight
        self.dist_weight = dist_weight
        if miner in ['BatchHardMiner']:
            self.miner = eval(miner)()
        else:
            self.miner = None

    def forward(self, embeddings, logits, labels):
        softmax_loss = 0
        center_loss = 0
        for key, center in self.centers.items():
            embs = embeddings[key]
            lgts = logits[key]
            center.update(embs, labels)
            softmax_loss = softmax_loss + cross_entropy(lgts, labels)
            if self.miner is None:
                center_loss = center_loss + mse_loss(embs, center.get_centers()[labels])
            else:
                center_loss = center_loss + self.mse_with_miner(embs, labels, center.get_centers()[labels])

        euclidean_loss = 0
        for key in embeddings.keys():
            if key == '0': pass
            euclidean_loss = euclidean_loss + mse_loss(embeddings['0'], embeddings[key])

        self.softmax_loss = softmax_loss / len(self.centers)
        self.center_loss = center_loss / len(self.centers)
        self.euclidean_loss = euclidean_loss
        self.total_loss = self.softmax_loss + self.center_weight * self.center_loss + self.dist_weight * self.euclidean_loss
        return self.total_loss

    def mse_with_miner(self, embeddings, labels, ref_embed):
        anchor_idx, pos_idx, neg_idx = self.miner(embeddings, labels, ref_embed, labels)
        return mse_loss(embeddings[anchor_idx], ref_embed[pos_idx]) \
            - mse_loss(embeddings[anchor_idx], ref_embed[neg_idx])
