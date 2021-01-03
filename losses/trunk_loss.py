import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn.functional import mse_loss

from .center_buffer import CenterBuffer
from miners import BatchHardMiner


class TrunkLoss(nn.Module):
    """
    Trunk Loss used in DCR

    Args:
        num_classes: (int) number of classses (in LRFR, number of persons)
        num_dimensions: (int) dimension of each class center,
                              should be consistent with the dimension of feature embeddings
        update_factor: (float) momentum factor used for center updating
        beta: (float) weight for center loss
    """

    def __init__(self,
                 num_classes,
                 num_dimensions,
                 update_factor=0.6,
                 beta=0.008,
                 trunk_miner=None,
                 trunk_within='center'):
        super(TrunkLoss, self).__init__()
        self.center = CenterBuffer(num_classes, num_dimensions, update_factor)
        self.beta = beta
        if trunk_miner in ["BatchHardMiner"]:
            self.miner = eval(trunk_miner)()
            self.miner_within = trunk_within
        else:
            self.miner = None

    def forward(self, embeddings, logits, labels):
        self.center.update(embeddings, labels)
        centers = self.center.get_centers()
        if self.miner is None:
            self.softmax_loss = cross_entropy(logits, labels, reduction='mean')
            self.center_loss = mse_loss(embeddings, centers[labels], reduction='mean')
            self.total_loss = self.softmax_loss + self.beta * self.center_loss
        else:
            self.softmax_loss = cross_entropy(logits, labels, reduction='mean')
            if self.trunk_within in ['center']:
                self.center_loss = self.mse_with_miner(embeddings, labels, centers[labels])
            elif self.trunk_within in ['embeddings']:
                self.center_loss = self.mse_with_miner(embeddings, labels, embeddings)
            elif self.trunk_within in ['none']:
                self.center_loss = 0
            self.total_loss = self.softmax_loss + self.beta * self.center_loss
        return self.total_loss

    def mse_with_miner(self, embeddings, labels, ref_embed):
        anchor_idx, pos_idx, neg_idx = self.miner(embeddings, labels, ref_embed, labels)
        # print(anchor_idx.shape, pos_idx.shape, neg_idx.shape)
        mse_pos = mse_loss(embeddings[anchor_idx], ref_embed[pos_idx], reduction='none')
        mse_neg = mse_loss(embeddings[anchor_idx], ref_embed[neg_idx], reduction='none')
        loss = nn.functional.relu(mse_pos - mse_neg + 0.05)

        return torch.mean(loss)
