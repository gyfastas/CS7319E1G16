import torch


__all__ = ['train_trunk', 'val_trunk']


def train_trunk(gpu, batch_data, model, criterion, optimizer, losses):
    images = batch_data['data']
    labels = batch_data['label'].long()

    if isinstance(images, torch.Tensor):
        images = {'0':images}

    # concatenate images of different resolutions along the batch dimension
    # images: [K,N,C,H,W]->[KN,C,H,W]
    num_scales = len(images)
    images = torch.cat(tuple(images.values()), dim=0)
    # repeat labels accordingly
    labels = labels.repeat(num_scales)

    if gpu is not None:
        images = images.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True)

    # compute output
    outputs = model(images, trunk_only=True)
    embeddings = outputs['trunk_embeddings']
    logits = outputs['trunk_logits']
    loss = criterion(embeddings, logits, labels)

    # record loss
    losses.update(loss.item(), images.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def val_trunk(gpu, batch_data, model, results=None):
    images = batch_data['data']
    labels = batch_data['label'].long()

    if isinstance(images, torch.Tensor):
        images = {'0':images}

    # concatenate images of different resolutions along the batch dimension
    # images: [K,N,C,H,W]->[KN,C,H,W]
    num_scales = len(images)
    images = torch.cat(tuple(images.values()), dim=0)

    if gpu is not None:
        images = images.cuda(gpu, non_blocking=True)

    # compute output
    with torch.no_grad():
        outputs = model(images, trunk_only=True)
    embeddings = outputs['trunk_embeddings'].detach().cpu()

    batch_embeds = dict(zip(batch_data['data'].keys(), embeddings.chunk(num_scales, dim=0)))
    batch_labels = labels.cpu()
    if results is not None:
        results.append((batch_embeds, batch_labels))
