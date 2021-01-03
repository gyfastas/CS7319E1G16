import math
import torch
import torch.nn as nn
import torch.distributed as dist

from utils import reduce_mean


# Tensor.mul_(Tensor) and Tensor.div_(Tensor) do not support inplace operation
def mul_inplace(tensor, index, value):
    target = tensor[index]
    target.mul_(value)
    tensor.index_copy_(0, index, target)

def div_inplace(tensor, index, value):
    target = tensor[index]
    target.div_(value)
    tensor.index_copy_(0, index, target)

class CenterBuffer(nn.Module):

    '''Implementation of centers in center loss'''

    def __init__(self, num_classes, num_dimensions, update_factor=0.6):
        super().__init__()
        self.update_factor = update_factor
        centers = torch.zeros(num_classes, num_dimensions).float()
        centers.normal_(0, math.sqrt(2. / num_classes))
        self.register_buffer('centers', centers)

    def get_centers(self):
        return self.centers.detach()

    @torch.no_grad()
    def update(self, embeddings, labels):
        if not self.training:
            pass
        centers = self.centers
        # calculate gradient (in the case of L2 loss) and decay it by momentum factor
        residual = torch.sub(embeddings, centers[labels]).mul(self.update_factor)
        # get classes whose centers should be updated
        labels_unique, labels_count = labels.unique(return_counts=True)
        # preprocess current centers for the subsequent averaging
        mul_inplace(centers, labels_unique, labels_count[:,None])
        # add gradient to centers of corresponding class
        centers.index_add_(0, labels.long(), residual)
        # average over the number of samples in each updated class
        div_inplace(centers, labels_unique, labels_count[:,None])
        # synchronize across all ranks in case of distributed training
        self.centers = reduce_mean(centers)