import torch


__all__ = ['train_branch', 'val_branch']


def train_branch(gpu, batch_data, model, criterion, optimizer, losses):
    images = batch_data['data']
    labels = batch_data['label'].long()

    # make sure the number of scales exceeds 1
    assert isinstance(images, dict) and len(images) > 1

    embeddings, logits = {}, {}
    for scale, imgs in images.items():
        scale = scale.split('_')[-1]
        if gpu is not None:
            imgs = imgs.cuda(gpu, non_blocking=True)
        # compute output
        outputs = model(imgs, idx=int(scale))
        embeddings[scale] = outputs['branch_embeddings']
        logits[scale] = outputs['branch_logits']

    fkey = list(embeddings.keys())[0]
    labels = labels.to(embeddings[fkey].device)
    loss = criterion(embeddings, logits, labels)
    losses.update(loss.item(), labels.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def val_branch(gpu, batch_data, model, results=None):
    images = batch_data['data']
    labels = batch_data['label'].long()

    # make sure the number of scales exceeds 1
    assert isinstance(images, dict) and len(images) > 1

    embeddings = {}
    for scale, imgs in images.items():
        if gpu is not None:
            imgs = imgs.cuda(gpu, non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(imgs, idx=int(scale.split('_')[-1]))
        embeddings[scale] = outputs['branch_embeddings'].detach().cpu()

    batch_embeds = embeddings
    batch_labels = labels.cpu()
    if results is not None:
        results.append((batch_embeds, batch_labels))
