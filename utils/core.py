import os
import math
import torch
import shutil
from .dist import broadcast, reduce_sum, collect_results_gpu, get_dist_info
from . import common_functions as c_f
from PIL import ImageFilter
import random


def load_model(model, state_dict, strict=False, init_branches=False):
    state_dict_copy = state_dict.copy()
    try:
        model.load_state_dict(state_dict_copy, strict=strict)
    except:
        for k in state_dict.keys():
            if len(k.split(".")) >= 2 and k.split(".")[-2] == "fc2":
                state_dict_copy.pop(k)
                print("Not loading {}".format(k))
        model.load_state_dict(state_dict_copy, strict=strict)
    if init_branches:
        for branch in model.branches:
            branch.fc1.bias.data.zero_()
            branch.fc1.weight.data.copy_(torch.diag(torch.ones(branch.fc1.in_features, dtype=torch.float)))
            branch.fc2.bias.data.copy_(model.trunk.fc2.bias.data)
            branch.fc2.weight.data.copy_(model.trunk.fc2.weight.data)
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirpath='experiments'):
    filename = os.path.join(dirpath, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(dirpath, 'model_best.pth.tar'))


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        fmtmsg = '\t'.join(entries)
        if self.logger is not None:
            self.logger.info(fmtmsg)
        else:
            print(fmtmsg)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    if args.schedule:
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

"""
Code from https://github.com/KevinMusgrave/pytorch-metric-learning
"""
class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=True):
        super().__init__()
        self.collect_stats = collect_stats

    def add_to_recordable_attributes(
        self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_f.add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )

    def reset_stats(self):
        c_f.reset_stats(self)


class BaseDistance(ModuleWithRecords):
    def __init__(
        self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(
        self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            with torch.no_grad():
                stats_dict = {
                    "initial_avg_query_norm": torch.mean(
                        self.get_norm(query_emb)
                    ).item(),
                    "initial_avg_ref_norm": torch.mean(self.get_norm(ref_emb)).item(),
                    "final_avg_query_norm": torch.mean(
                        self.get_norm(query_emb_normalized)
                    ).item(),
                    "final_avg_ref_norm": torch.mean(
                        self.get_norm(ref_emb_normalized)
                    ).item(),
                }
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype).to(device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class ModuleWithRecordsAndDistance(ModuleWithRecords):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2)

    def get_distance(self):
        return self.get_default_distance()