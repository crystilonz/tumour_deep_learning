import os
from os.path import isdir

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.interface.LungRNN import LungRNN
from models.LSTM import LSTM
from data_manipulation.lung_caption_dataset import LungCaptionDataset
from data_manipulation.lung_caption_vocab import Vocabulary

import os
import pickle
import argparse
import functools
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap

from utils.plotting import plot_loss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group(backend='nccl',
                            rank=rank,
                            world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def rnn_distributed_train_step(model: torch.distributed.fsdp.FullyShardedDataParallel,
                               rank,
                               world_size,
                               dataloader: DataLoader,
                               optimizer: torch.optim.Optimizer,
                               epoch,
                               sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)

    for feature, caption in dataloader:
        feature, caption = feature.to(rank), caption.to(rank)
        cap_preds = model(feature, caption)

        # loss
        batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
        loss = F.cross_entropy(batch_vocab_dim, caption, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(feature)  # number of samples in batch

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    avg_loss = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        print(f'Training Loss: {avg_loss:.4f}')

    return avg_loss


def rnn_distributed_validate_step(model: torch.distributed.fsdp.FullyShardedDataParallel,
                                  rank,
                                  world_size,
                                  dataloader: DataLoader):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)

    with torch.no_grad():
        for feature, caption in dataloader:
            feature, caption = feature.to(rank), caption.to(rank)
            cap_preds = model(feature, caption)

            # loss
            batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
            loss = F.cross_entropy(batch_vocab_dim, caption, reduction='sum')

            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(feature)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        print(f'Validation Loss: {avg_loss:.4f}')

    return avg_loss


def train_rnn_distributed(rank, world_size, args):
    # # dataset args
    # dataset_kwargs = {'data_dir': args.data_dir,
    #                   'vocab_dir': args.vocab_dir,
    #                   }
    #
    # # model args
    # model_kwargs = {'model': args.models,
    #                 }
    #
    # # args for training
    # training_kwargs = {'batch_size': args.batch_size,
    #                    }
    #
    # # args for validating
    # validation_kwargs = {'batch_size': args.validate_batch_size,
    #                      }
    #
    # # args for distributed training
    # cuda_kwargs = {'num_workers': args.num_workers,
    #                'pin_memory': args.pin_memory,
    #                'shuffle': args.shuffle}
    #
    # # args for saving
    # saving_kwargs = {'save_dir': args.save_dir,}

    with open(args.pkl, 'rb') as fp:
        split_dict = pickle.load(fp)

    training_dataset = split_dict['training_dataset']
    validating_dataset = split_dict['validating_dataset']
    # testing_dataset = split_dict['testing_dataset']
    collate_fn = split_dict['collate_fn']

    training_sampler = DistributedSampler(training_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    validating_sampler = DistributedSampler(validating_dataset, rank=rank, num_replicas=world_size)
    # testing_sampler = DistributedSampler(testing_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.train_batch_size, 'sampler': training_sampler, }
    valid_kwargs = {'batch_size': args.validate_batch_size, 'sampler': validating_sampler, }
    # test_kwargs = {'batch_size': args.test_batch_size, 'sampler': testing_sampler, }

    cuda_kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    valid_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(dataset=training_dataset,
                              shuffle=True,
                              collate_fn=collate_fn,
                              **train_kwargs)
    validate_loader = DataLoader(dataset=validating_dataset,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 **valid_kwargs)
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    vocab = Vocabulary()
    vocab.load(args.vocab_dir)

    model = LSTM(input_size=args.input_size,
                 embed_size=args.embed_size,
                 hidden_size=args.hidden_size,
                 vocab=vocab,
                 num_layers=args.num_layers, ).to(rank)
    model = FSDP(model,
                 auto_wrap_policy=auto_wrap_policy, )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    init_start_event.record()

    training_loss = []
    validating_loss = []

    for epoch in range(0, args.epochs):
        t_loss = rnn_distributed_train_step(model=model,
                                            rank=rank,
                                            world_size=world_size,
                                            dataloader=train_loader,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            sampler=training_sampler)
        v_loss = rnn_distributed_validate_step(model=model,
                                               rank=rank,
                                               world_size=world_size,
                                               dataloader=validate_loader)

        training_loss.append(t_loss)
        validating_loss.append(v_loss)

    init_end_event.record()

    if rank == 0:
        # elapsed time
        init_end_event.synchronize()
        print(f'CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000 :.2f} s.')
        print("Model: ")
        print(f'{model}')

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)

            torch.save(states, os.path.join(args.save_dir, 'model.pt'))

            # draw loss
            plot_loss(training_loss, validating_loss,
                      save_to=os.path.join(args.save_dir, 'loss.png'),
                      show_plot=False)
