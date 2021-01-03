#!/usr/bin/env python
import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import core
import utils as tools
from logger import Logger
from config import parser, dataset_classes, model_classes
from losses import TrunkLoss, CMLoss


def main():
    args = parser.parse_args()
    args.timestamp = tools.get_timestamp()

    tools.mkdir_or_exist(args.workdir)
    tools.setup(args.benchmark, args.deterministic, args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    logger = Logger(args)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, **kwargs):
            pass
        logger.info = print_pass
    logger.init_info()
    logger.info("Configuration:\n{}".format(json.dumps(args.__dict__, indent=4)))

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    if args.aug_plus:
        train_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([tools.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.rgb_mean, std=args.rgb_std)
        ])
    else:
        train_augmentation = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.rgb_mean, std=args.rgb_std)
        ])
    val_augmentation = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.rgb_mean, std=args.rgb_std)
    ])
    train_dataset = dataset_classes[args.dataset](args.data, subset=args.subset,
        mode='train', image_transform=train_augmentation, scale_size=args.scale_list)
    if args.val_dataset is None:
        args.val_dataset = args.dataset
    if args.val_scale_list is None:
        args.val_scale_list = args.scale_list
    val_dataset = dataset_classes[args.val_dataset](args.data, subset=args.subset,
        mode='val', image_transform=val_augmentation, scale_size=args.val_scale_list, task=args.task)

    # create model
    logger.info("=> creating model ..")
    model = model_classes[args.stage](
            backbone=args.backbone,
            layers=args.layers,
            out_channels=args.out_channels,
            mid_channels=args.mid_channels,
            num_scales=train_dataset.num_scales,
            num_classes=train_dataset.num_classes,
            normalized_embeddings=args.normalized_embeddings,
            use_bn=not args.wobn)
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(str(model))

    # freeze part of model and create loss function
    if args.stage == 'trunk':
        # freeze branches
        for param in model.get_branches().parameters():
            param.requires_grad = False
        criterion = TrunkLoss(train_dataset.num_classes, args.out_channels,
                              args.center_update_factor, args.trunkloss_factor,
                              args.trunkloss_miner, args.trunkloss_miningin)
    elif args.stage == 'branch':
        # freeze trunk
        if not args.no_fixed_trunk:
            for param in model.get_trunk().parameters():
                param.requires_grad = False
        criterion = CMLoss(train_dataset.num_classes, args.out_channels, train_dataset.num_scales,
                           args.center_update_factor, args.centerloss_factor, args.distloss_factor, args.branchloss_miner)
    else:
        criterion = SIDLoss(train_dataset.num_classes, args.out_channels,
                           args.center_update_factor, args.trunkloss_factor, args.trunkloss_miner,
                           sidloss_factor=args.sidloss_factor)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            criterion.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            criterion.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
    else:
        raise NotImplementedError("Only CUDA or DistributedDataParallel is supported.")

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), args.lr)
    monitor = tools.EarlyStopMonitor(mode=args.es_mode, patience=args.patience)

    if args.load:
        if os.path.isfile(args.load):
            logger.info("=> loading checkpoint '{}'".format(args.load))
            if args.gpu is None:
                checkpoint = torch.load(args.load)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.load, map_location=loc)

            model = tools.load_model(model, checkpoint['state_dict'],
                strict=False, init_branches=args.init_branches)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.load))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> resume checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            criterion.load_state_dict(checkpoint['criterion'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            monitor.load_state_dict(checkpoint['monitor'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    elif args.Msampler:
        train_sampler = tools.MPerClassSampler(train_dataset.label_list, 4, len(train_dataset))
        test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False)

    # get function handle of batch processor
    batch_processor_train = getattr(core, 'train_{}'.format(args.stage))
    batch_processor_val = getattr(core, 'val_{}'.format(args.stage))

    metric = core.val(batch_processor_val, val_loader, model, 0, args, logger)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        tools.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        core.train(batch_processor_train, train_loader, model, criterion, optimizer, epoch, args, logger)
        metric = core.val(batch_processor_val, val_loader, model, epoch, args, logger)

        if tools.get_dist_info()[0] == 0:
            tools.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'monitor': monitor.state_dict(),
            }, is_best=monitor.is_best(metric),
               filename='checkpoint_{:04d}.pth.tar'.format(epoch),
               dirpath=args.workdir)

        # metric must be d across all ranks
        if monitor.check(metric):
            logger.info("No improvement is seen for {} epoches. '\
                'Training is terminated.".format(monitor.patience))
            break
        tools.barrier()


if __name__ == '__main__':
    main()
