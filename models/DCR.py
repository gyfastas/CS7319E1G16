import torch
import torch.nn as nn

from .trunk import TruncatedResNet, DCRTrunk, NewDCRTrunk
from .branch import DCRBranch


class DCR(nn.Module):

    def __init__(self,
                 backbone,
                 num_scales,
                 num_classes,
                 layers=[1, 2, 5, 3],
                 out_channels=512,
                 mid_channels=512,
                 normalized_embeddings=False,
                 use_bn=True):
        super().__init__()
        if backbone == 'DCRTrunk':
            self.trunk = DCRTrunk(layers=layers, num_classes=num_classes, out_channels=out_channels,
                                  normalized_embeddings=normalized_embeddings)
        elif backbone == 'NewDCRTrunk':
            self.trunk = NewDCRTrunk(layers=layers, num_classes=num_classes, out_channels=out_channels,
                                     normalized_embeddings=normalized_embeddings, use_bn=use_bn)
        elif backbone.startswith('ResNet'):
            assert len(backbone.split('-')) == 2, 'Number of layers should be specified as `ResNet-n`'
            depth = backbone.split('-')[-1]
            self.trunk = TruncatedResNet(depth, num_classes=num_classes, out_channels=out_channels,
                                         normalized_embeddings=normalized_embeddings)
        else:
            raise "TODO"
        self.branches = nn.ModuleList([
            DCRBranch(num_classes, out_channels, mid_channels, 
                normalized_embeddings=normalized_embeddings) for i in range(num_scales)])

    def get_trunk(self):
        return self.trunk

    def get_branches(self):
        return self.branches

    def forward(self, img, idx=0, trunk_only=False):
        trunk_embeddings, trunk_logits = self.trunk(img, not trunk_only)
        if not trunk_only:
            branch_embeddings, branch_logits = self.branches[idx](trunk_embeddings)
        else:
            branch_embeddings, branch_logits = None, None
        return {
            'trunk_embeddings': trunk_embeddings,
            'trunk_logits': trunk_logits,
            'branch_embeddings': branch_embeddings,
            'branch_logits': branch_logits
        }
