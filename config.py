import argparse
from datasets import SCfaceDataset, CasiaDataset, LFWDataset, DebugDataset
from models import DCR

model_classes = dict(
    trunk=DCR,
    branch=DCR,
)
dataset_classes = dict(
    SCface=SCfaceDataset,
    Casia=CasiaDataset,
    LFW=LFWDataset,
    Debug=DebugDataset)

parser = argparse.ArgumentParser(description='CS7319G16-Project')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('workdir', help='path to save experiment')
parser.add_argument('stage', choices=('trunk', 'branch'), help='stage to run')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--benchmark', default=True, type=bool,
                    help='cudnn.benchmark (default: True)')
parser.add_argument('--deterministic', default=False, type=bool,
                    help='cudnn.deterministic (default: False)')

# dataset
parser.add_argument('--dataset', default='Casia', choices=tuple(dataset_classes.keys()),
                    help='name of the dataset (default: Casia)')
parser.add_argument('--val-dataset', default=None, choices=tuple(dataset_classes.keys()),
                    help='name of the validation dataset (default: the same as `dataset`)')
parser.add_argument('--task', default='R', choices=('R', 'V', 'IR', 'IV'),
                    help='task for validation, retrival (R) or verification (V) (default: R)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--subset', default=0, type=int,
                    help='id of subset to use (default: 0)')
parser.add_argument('--scale-list', default=[8, 12, 16, 20], type=int, nargs='+',
                    help='LR scales used in training. Omitted in SCfaceDataset (default: [8,12,16,20])')
parser.add_argument('--val-scale-list', default=None, type=int, nargs='+',
                    help='scale-list for the validation dataset (default: the same as `scale-list`)')
parser.add_argument('--img-size', default=(120,120), type=int, nargs='+',
                    help='size of input images (default: (120,120))')
parser.add_argument('--rgb-mean', default=[0.5, 0.5, 0.5], type=int, nargs='+',
                    help='mean rgb values over dataset')
parser.add_argument('--rgb-std', default=[0.5, 0.5, 0.5], type=int, nargs='+',
                    help='std rgb values over dataset')
parser.add_argument('--Msampler', action='store_true', help='Using MPerClass Sampler')
parser.add_argument('--aug-plus', action='store_true', 
                    help='Use powerful training augmentation (see MocoV2)')

# model
parser.add_argument('--backbone', default='DCRTrunk', type=str,
                    help='backbone network (default: DCRTrunk)')
parser.add_argument('--layers', default=(1, 2, 5, 3), type=int, nargs='+',
                    help='number of layers for each stage in DCRTrunk (default: [1,2,5,3])')
parser.add_argument('--out-channels', default=512, type=int,
                    help='number of output dimension of backbone (also the number '
                    'of input channels of branch network) (default: 512)')
parser.add_argument('--mid-channels', default=512, type=int,
                    help='number of intermediate channels of branch network (default: 512)')
parser.add_argument('--normalized_embeddings', action='store_true',
                    help='whether to use L2 normalized embeddings')
parser.add_argument('--wobn', action='store_true',
                    help='whether to omit bn in backbone')
parser.add_argument('--center-update-factor', default=0.6, type=float,
                    help='momentum factor of center updating (default: 0.6)')
parser.add_argument('--trunkloss-factor', default=0.008, type=float,
                    help='rescale factor of trunk loss (default: 0.008)')
parser.add_argument('--centerloss-factor', default=0.008, type=float,
                    help='rescale factor of center loss (default: 0.008)')
parser.add_argument('--distloss-factor', default=0.008, type=float,
                    help='rescale factor of distance loss (default: 0.008)')
parser.add_argument('--load', default="", type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default="", type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--init-branches', action='store_true',
                    help='initialize branches with weights from trunk')
parser.add_argument('--trunkloss-miner', default="", type=str,
                    help='miner for center loss in trunkloss, (default:none)')
parser.add_argument('--trunkloss-miningin', default="center", type=str,
                    choices=('center', 'embedding', 'none'),help='miner for center loss in trunkloss, (default:center)')
parser.add_argument('--branchloss-miner', default="", type=str,
                    help='miner for center loss in trunkloss, (default:none)')

# optimizer
parser.add_argument('--optimizer', default='Adam', type=str,
                    choices=('SGD', 'Adam'), help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=None, nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of maximum epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--patience', default=0, type=int,
                    help='teminate training if no improvement is seen '
                    'for a patient number of epoches. If disabled, set to 0 (default: 0)')
parser.add_argument('--es-mode', default='max', type=str, choices=['min', 'max'],
                    help='early stopping mode')
parser.add_argument('--fixed-BN', action='store_true',
                    help='whether fix BN (using model.eval()) while finetuning')
parser.add_argument('--no-fixed-trunk', '-nft', action='store_true',
                    help='Do not fix trunk when training branch')

# distributed
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# misc
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
