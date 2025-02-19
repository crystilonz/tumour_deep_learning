import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Type
import time
import tqdm
from datetime import datetime

from models.interface.LungRNN import LungRNN
from models.LSTM import LSTM
from data_manipulation.lung_caption_dataset import LungCaptionDataset
from data_manipulation.lung_caption_vocab import Vocabulary
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from utils.plotting import plot_loss

import pickle



def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl",
                            rank=rank,
                            world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_model(args):
    vocab = Vocabulary()
    vocab.load(args.vocab_dir)
    
    model = LSTM(input_size=args.input_size,
                 embed_size=args.embed_size,
                 hidden_size=args.hidden_size,
                 vocab=vocab,
                 num_layers=args.num_layers)
    
    return model, vocab

def train(args, model, device, rank, world_size, dataloader, optimizer, epoch, sampler=None):
    model.train()
    # local_rank = int(os.environ['LOCAL_RANK'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    
    fsdp_loss = torch.zeros(2).to(device)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(dataloader)), colour="blue", desc="r0 Training Epoch"
        )
    for feature, caption in dataloader:
        feature, caption = feature.to(device), caption.to(device)
        cap_preds = model(feature, caption)

        # loss
        batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
        loss = F.cross_entropy(batch_vocab_dim, caption, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(feature)  # number of samples in batch
        
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, device, rank, world_size, dataloader):
    model.eval()
    # local_rank = int(os.environ['LOCAL_RANK'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    fsdp_loss = torch.zeros(2).to(device)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(dataloader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for feature, caption in dataloader:
            feature, caption = feature.to(device), caption.to(device)
            cap_preds = model(feature, caption)

            # loss
            batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
            loss = F.cross_entropy(batch_vocab_dim, caption, reduction='sum')

            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(feature)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def fsdp_main(args):

    model, vocab = setup_model(args)

    # local_rank = int(os.environ['LOCAL_RANK'])
    # rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    # print('local rank', os.environ['SLURM_LOCALID'])
    # print('world rank', os.environ['SLURM_PROCID'])
    # print('world size', os.environ['SLURM_NTASKS'])
    
    # print('current device index:', torch.cuda.current_device())
    # print('uuid', torch.cuda._raw_device_count_nvml())
    # print('device name:', torch.cuda.get_device_name())
    # print('available gpus:', torch.cuda.device_count())


    with open(args.pkl, 'rb') as fp:
        split_dict = pickle.load(fp)

    training_dataset = split_dict['training_dataset']
    validating_dataset = split_dict['validating_dataset']
    # testing_dataset = split_dict['testing_dataset']
    collate_fn = split_dict['collate_fn']

    
    training_sampler = DistributedSampler(training_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    validating_sampler = DistributedSampler(validating_dataset, rank=rank, num_replicas=world_size)
    # testing_sampler = DistributedSampler(testing_dataset, rank=rank, num_replicas=world_size)

    setup(rank, world_size)


    train_kwargs = {'batch_size': args.train_batch_size, 'sampler': training_sampler}
    valid_kwargs = {'batch_size': args.validate_batch_size, 'sampler': validating_sampler}
    cuda_kwargs = {'num_workers': args.num_workers,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    valid_kwargs.update(cuda_kwargs)
    

    train_loader = DataLoader(dataset=training_dataset,
                              collate_fn=collate_fn,
                              **train_kwargs)
    validate_loader = DataLoader(dataset=validating_dataset,
                                 collate_fn=collate_fn,
                                 **valid_kwargs)

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    # torch.cuda.device(local_rank)
    current_device = torch.device('cuda')  # should work


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    # bf16_ready = (
    # torch.version.cuda
    # and torch.cuda.is_bf16_supported()
    # and LooseVersion(torch.version.cuda) >= "11.0"
    # and dist.is_nccl_available()
    # and nccl.version() >= (2, 10)
    # )

    # if bf16_ready:
    #     mp_policy = bfSixteen
    # else:
    #     mp_policy = None # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=auto_wrap_policy,
        # mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    if rank == 0:
        # time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    # if rank == 0 and args.track_memory:
    #     mem_alloc_tracker = []
    #     mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, current_device, rank, world_size, train_loader, optimizer, epoch, sampler=training_sampler)
        curr_val_loss = validation(model, current_device, rank, world_size, validate_loader)
        # optimizer.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            # if args.run_validation:
            val_acc_tracking.append(curr_val_loss.item())

            # if args.track_memory:
            #     mem_alloc_tracker.append(
            #         format_metrics_to_gb(torch.cuda.memory_allocated())
            #     )
            #     mem_reserved_tracker.append(
            #         format_metrics_to_gb(torch.cuda.memory_reserved())
            #     )
            print(f"completed save and stats zone...")

    
        if args.save_model and curr_val_loss < best_val_loss:
            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            #print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> saving model ...")
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir, exist_ok=True)

                torch.save(cpu_state, os.path.join(args.save_dir, 'model.pt'))
            
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                if rank==0:
                    print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    if rank == 0:
        print(f"--> finishing")
        plot_loss(train_acc_tracking, val_acc_tracking,
                save_to=os.path.join(args.save_dir, 'loss.png'),
                show_plot=False)
    cleanup()