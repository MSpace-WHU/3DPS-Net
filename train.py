import os
import math
import torch
import argparse
from models.dataset import ForestDataset, ForestDataset_collate_fn
from models.network import PSNet3d
from torch.utils.data import DataLoader
from functools import partial
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from models.utils import processbar, Logger
from torch.utils.tensorboard import SummaryWriter
import pdb


def update_lr(base_lr, cur_epoch, epoch, optimizer, warm_up_epoch=0, min_lr=0.00005):
    if cur_epoch < warm_up_epoch:
        lr = base_lr * cur_epoch / warm_up_epoch
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (cur_epoch - warm_up_epoch) / (epoch - warm_up_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print("lr update finished  cur lr: %.8f" % lr)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./train.py --dataset_name=forest
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--dataset_name', default='forest', type=str, help='The name of folder')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model path')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    device = torch.device(args.local_rank)

    # configuration
    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_dir, 'params', args.dataset_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'experiment'))
    logger = Logger(args.dataset_name, os.path.join(save_path, 'log_train.txt'))

    # dataset_configuration
    train_dataset = ForestDataset(root_dir,
                                  args.dataset_name,
                                  split='trainval',
                                  grid_size=8.5)

    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=dist.get_world_size(),
                                       rank=args.local_rank,
                                       shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=8,
                              collate_fn=partial(ForestDataset_collate_fn))

    # network_configuration
    network = PSNet3d().cuda(args.local_rank)
    network = DistributedDataParallel(network,
                                      device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)
    if args.pretrained:
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(args.pretrained, map_location='cpu').items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

        # network.load_state_dict(torch.load(args.pretrained)) if os.path.exists(args.pretrained) else None

    network.train()

    # Hyperparameters
    base_lr = 0.0001
    batch_size = 4
    epochs = 10000

    # Optimizer
    optimizer = torch.optim.AdamW(network.parameters(), betas=(0.9, 0.999), lr=base_lr, eps=1e-8, weight_decay=1.0e-3)

    # Loss
    loss_fn = torch.nn.BCELoss()

    max_acc = 0
    for epoch in range(1, epochs + 1):
        loss_val = 0
        processed = 0
        correct_point_num = 0
        totle_point_num = 0

        for batch_idx, (pcd_list, prompt_ind_list, labels) in enumerate(train_loader):

            labels = labels.float().to(device)
            feats_list = network(pcd_list, prompt_ind_list, device)

            pred = torch.cat(feats_list, dim=0).view(-1)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if torch.distributed.get_rank() == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch)

            loss_val += loss.item()
            processed += len(pcd_list)

            pred = pred.round()
            correct_point_num += torch.sum(pred == labels, dim=0).item()
            totle_point_num += labels.shape[0]
            if rank == 0:
                print("\rprocess: %s  loss: %.5f  acc: %.5f" % (
                    processbar(processed, len(train_dataset)), loss.item(), correct_point_num / totle_point_num),
                      end="")

        acc = correct_point_num / totle_point_num
        if rank == 0:
            logger.write("\nepoch: %d  train finish, loss: %.5f  acc: %.5f" % (epoch, loss_val, acc))
        val_loss = loss_val
        if max_acc < acc:
            max_acc = acc
            if rank == 0:
                logger.write("save ... ")
            torch.save(network.state_dict(), os.path.join(save_path, 'epoch_{}_{:7.5f}.pth'.format(epoch, acc)))
            if rank == 0:
                logger.write("save finish !")

        update_lr(base_lr, epoch, epochs, optimizer)
