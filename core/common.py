import time
import json
from tqdm import tqdm

import torch

import utils as tools


def train(batch_processor, train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = tools.AverageMeter('Time', ':6.3f')
    data_time = tools.AverageMeter('Data', ':6.3f')
    losses = tools.AverageMeter('Loss', ':.4e')
    progress = tools.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger)

    # fix BN
    if args.fixed_BN:
        model.eval()
    else:
        model.train()
    criterion.train()
    end = time.time()
    for i, batch_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_processor(args.gpu, batch_data, model, criterion, optimizer, losses)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def val(batch_processor, val_loader, model, epoch, args, logger):
    results = []

    # switch to val mode
    model.eval()
    for i, batch_data in enumerate(tqdm(val_loader, desc='Evaluation')):
        batch_processor(args.gpu, batch_data, model, results)

    # collect results and get evaluation metric
    metrics, early_stop_criterion = evaluate(results, torch.tensor([i + 1], device='cuda'),
                                                   val_loader.dataset.evaluate)

    if tools.get_dist_info()[0] == 0:
        logger.info('[Evaluation] - Epoch: [{}]'.format(epoch))
        logger.info('Evalution metrics:\n{}'.format(json.dumps(metrics, indent=4)))
    return early_stop_criterion


def evaluate(results, num_batches, evaluate_fn):
    num_batches = tools.reduce_sum(num_batches)
    results = tools.collect_results_gpu(results, num_batches.item())

    rank = tools.get_dist_info()[0]
    if rank == 0:
        all_embeds = list(map(lambda r:r[0], results))
        all_labels = list(map(lambda r:r[1], results))
        all_embeds = {k:torch.cat(tuple(map(lambda r:r[k], all_embeds)), dim=0) for k in all_embeds[0].keys()}
        all_labels = torch.cat(all_labels, dim=0)
        metrics = evaluate_fn(all_embeds, all_labels)
        early_stop_criterion = torch.tensor([metrics['criterion']], device='cuda')
    else:
        metrics = None
        early_stop_criterion = torch.tensor([0.], device='cuda')

    # early_stop_criterion is used for all ranks, so broadcast it
    early_stop_criterion = tools.broadcast(early_stop_criterion, 0)
    return metrics, early_stop_criterion