#!/usr/bin/env python
import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision.transforms as transforms

from models import DCR
from logger import Logger
from datasets import SCfaceDataset, CasiaDataset, LFWDataset, DebugDataset
import utils as tools

dataset_classes = dict(
    SCface=SCfaceDataset,
    Casia=CasiaDataset,
    LFW=LFWDataset,
    Debug=DebugDataset)

parser = argparse.ArgumentParser(description='CS7319G16-Project')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('config', help='path to config file')
parser.add_argument('load', type=str, help='path to latest checkpoint')
parser.add_argument('stage', type=str, default='trunk', choices=['trunk', 'branch'])
parser.add_argument('task', type=str, default='R', choices=['R','V', 'IV', 'IR'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use (default: 0)')
parser.add_argument('--dataset', default='Casia', choices=tuple(dataset_classes.keys()),
                    help='name of the dataset (default: Casia)')

def main():
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args_old = json.load(f)
    args_old.update(args.__dict__)
    args.__dict__ = args_old
    args.timestamp = tools.get_timestamp()

    tools.mkdir_or_exist(args.workdir)
    tools.setup(True, False, None)
    
    logger = Logger(args, 'test_report.txt', mode='w')
    logger.init_info(save_config=False)
    logger.info("Configuration:\n{}".format(json.dumps(args.__dict__, indent=4)))
    
    # create dataset
    test_augmentation = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.rgb_mean, std=args.rgb_std)
    ])
    test_dataset = dataset_classes[args.dataset](args.data,
        subset=args.subset, mode='test', image_transform=test_augmentation,
        scale_size=args.scale_list, task=args.task)

    # create model
    logger.info("=> creating model ..")
    model = DCR(
        backbone=args.backbone,
        layers=args.layers,
        out_channels=args.out_channels,
        mid_channels=args.mid_channels,
        num_scales=test_dataset.num_scales,
        num_classes=test_dataset.num_classes,
        normalized_embeddings = args.normalized_embeddings)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    if args.load:
        if os.path.isfile(args.load):
            logger.info("=> loading checkpoint '{}'".format(args.load))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.load, map_location=loc)
            tools.load_model(model, checkpoint['state_dict'])
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.load))
    else:
        logger.warning("args.load is not specified. Model will be evaluated in random initialized.")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    results = []

    # switch to val mode
    model.eval()
    all_embeds = list()
    all_labels = list()
    for i, batch_data in enumerate(tqdm(test_loader)):
        images = batch_data['data']
        labels = batch_data['label'].long()
        all_labels.append(labels)

        if isinstance(images, torch.Tensor):
            images = {'0':images}
        # compute output
        embeddings = {}
        with torch.no_grad():
            embeddings = dict()
            for scale, imgs in images.items():
                if args.gpu is not None:
                    imgs = imgs.cuda(args.gpu, non_blocking=True)
                if args.stage=='trunk':
                    outputs = model(imgs, trunk_only=True)
                    embeddings[scale] = outputs['trunk_embeddings'].detach().cpu()
                elif args.stage=='branch':
                    outputs = model(imgs, idx=int(scale.split('_')[-1]))
                    embeddings[scale] = outputs['branch_embeddings'].detach().cpu()
        all_embeds.append(embeddings)
    # get evaluation metric
    all_embeds = {k:torch.cat(tuple(map(lambda r:r[k], all_embeds)), dim=0) for k in all_embeds[0].keys()}
    all_labels = torch.cat(all_labels, dim=0)
    metrics = test_loader.dataset.evaluate(all_embeds, all_labels)
    logger.info('Evalution metrics:\n{}'.format(json.dumps(metrics, indent=4)))


if __name__ == '__main__':
    main()
