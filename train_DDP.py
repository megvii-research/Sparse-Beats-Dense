#!/usr/bin/env python3
import argparse
from io import BytesIO
import os
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
from pathlib import Path
import sys
import time
import traceback
from loguru import logger
logger.remove()
if(LOCAL_RANK == 0):
    logger.add(sys.stdout, colorize=True, level="INFO", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
else:
    logger.add(sys.stderr, colorize=True, level="ERROR", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Network
import dataset
from utils import (
    TrainClock,
    log_rate_limited,
)


def ensure_dir(path: Path):
    """create directories if *path* does not exist"""

    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


class config(dataset.conf):
    # # Override
    # batch_size = 64

    base_lr = 3e-3
    epoch_num = 30
    checkpoint_interval = 1
    log_interval = 20

    exp_dir = os.path.dirname(__file__)
    exp_name = os.path.basename(exp_dir)
    local_train_log_path = './train_log'
    log_dir = str(local_train_log_path)
    log_model_dir = os.path.join(local_train_log_path, 'models')
    
    params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 4,
          'persistent_workers': True}


class Session:

    def __init__(self, config, net=None, rank=0, local_rank=0):
        self.log_dir = config.log_dir
        ensure_dir(self.log_dir)
        self.model_dir = config.log_model_dir
        ensure_dir(self.model_dir)

        self.clock = TrainClock()
        self.config = config
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.net = net
        self.optimizer: torch.optim.Optimizer = None
        self.rank = rank
        self.local_rank = local_rank
        self.task = None

    def start(self):
        self.save_checkpoint('start')

    def save_checkpoint(self, name):
        if self.rank != 0:
            return
        net = self.net.module if isinstance(self.net, DDP) else self.net
        net_state = net.state_dict()

        ckp = {
            'network': net_state,
            'clock': self.clock.make_checkpoint(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        config = self.config

        torch.save(ckp, Path(config.log_model_dir) / (name+'.ckpt'))

        # model-specific checkpoint
        ckp = {
            "network": net_state,
        }
        torch.save(ckp, Path(config.log_model_dir) / (name+'.net.ckpt'))


    def load_misc_checkpoint(self, ckp_path:Path):
        checkpoint = torch.load(
            ckp_path, 
            map_location=torch.device(f"cuda:{self.local_rank}")
        )
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    def load_net_state_dict(self, ckp_path:Path):
        if self.rank == 0:
            checkpoint = torch.load(
                ckp_path, 
                map_location=torch.device(f"cuda:{self.local_rank}")
            )
            self.net.load_state_dict(checkpoint['network'], strict=False)


def main():
    parser = argparse.ArgumentParser()
    default_devices = '*' if os.environ.get('RLAUNCH_WORKER') else '0'
    parser.add_argument('-d', '--device', default=default_devices)
    parser.add_argument('--fast-run', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('-r', '--restart', action='store_true')
    args = parser.parse_args()

    if(LOCAL_RANK == 0):
        log_path = Path(config.log_dir) / "worklog.log"
        logger.add(str(log_path.resolve()), colorize=True, level="INFO", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
    else:
        log_path = Path(config.log_dir) / f"worklog_{RANK}.log"
        logger.add(str(log_path.resolve()), colorize=True, level="ERROR", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    net = Network()
    try:
        if RANK == 0:
            # TODO: finetune
            net = net.cuda(LOCAL_RANK)
        else:
            net = net.cuda(LOCAL_RANK)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
    
    # create session
    sess = Session(config, net=net, rank=RANK, local_rank=LOCAL_RANK)
    clock = sess.clock
    
    continue_path = None
    if args.restart: # 
        continue_path = Path(os.path.join(config.log_model_dir, "latest"))
    elif continue_path is not None:
        continue_path = None

    net_continue_path = continue_path.with_name(continue_path.name+".net.ckpt") if continue_path else None
    if net_continue_path and os.path.exists(net_continue_path) and RANK == 0:
        sess.load_net_state_dict(net_continue_path)

    torch.distributed.barrier() # 所有进程等待rank=0进程load模型
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Using DDP train Model!")
        net = DDP(sess.net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
        sess.net = net

    datasets = dataset.Vidar()
    train_ds = torch.utils.data.DataLoader(datasets, **sess.config.params)
    
    opt = torch.optim.AdamW(sess.net.parameters(), lr=1., weight_decay=4e-8)
    total_step = len(train_ds) * sess.config.epoch_num
    base_lr = config.base_lr
    def customer_lr_func(step):
        return base_lr * (np.cos(step / total_step * np.pi) + 1) * 0.5 + 1e-3

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, customer_lr_func)

    sess.optimizer = opt
    sess.lr_scheduler = lr_scheduler

    # restore checkpoint
    if continue_path:
        misc_continue_path = continue_path.with_name(continue_path.name+".ckpt") if continue_path else None
        if misc_continue_path and os.path.exists(misc_continue_path):
            sess.load_misc_checkpoint(misc_continue_path)
    
    sess.start()
    log_output = log_rate_limited(min_interval=1)(logger.info)
    
    step_start  = clock.step
    loss_record, monitors_record = 0, {}

    time_train_start = time.time()
    for epoch in range(sess.config.epoch_num):
        net.train()
        time_iter_start = time.time()
        for idx, mini_batch_data in enumerate(train_ds):   
            tdata = time.time() - time_iter_start
            img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi = mini_batch_data
            mini_batch_data = {'img': img,
                               'radar': radar,
                               'radar_pts': padding_radar_pts,
                               'valid_radar_pts_cnt': valid_radar_pts_cnt,
                               'label': lidar,
                               'label_mask': lidar_mask,
                               'seg_mask_roi': seg_mask_roi,
                               }
            
            try:
                loss, monitors = net.module.forward_train(mini_batch_data)
            except Exception as e:
                traceback.print_exc()
                sys.exit(1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            time_train_passed = time.time() - time_train_start
            step_passed = clock.step - step_start
            eta = (total_step - clock.epoch) * 1.0 / max(step_passed, 1e-7) * time_train_passed
            time_iter_passed = time.time() - time_iter_start

            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
            loss_record += loss.item()
            if RANK == 0:
                loss_record += loss.item()
                if monitors:
                    for k,v in monitors.items(): 
                        monitors_record[k] = monitors_record.setdefault(k, 0) + v
                
                log_interval = config.log_interval # 每个epoch至少log一次
                if idx and (idx+1) % log_interval == 0:
                    
                    loss_record /= log_interval
                    if monitors_record:
                        for k,v in monitors_record.items(): 
                            monitors_record[k] /= log_interval

                    # print text info
                    meta_info = list()
                    meta_info.append('{:.2g} b/s'.format(1. / time_iter_passed))
                    meta_info.append('passed:{}'.format(format_time(time_train_passed)))
                    meta_info.append('eta:{}'.format(format_time(eta)))
                    meta_info.append('data_time:{:.2%}'.format(tdata / time_iter_passed))
                    meta_info.append('lr:{:.5g}'.format(lr))
                    meta_info.append('[{}:{}/{}]'.format(clock.epoch, idx+1, len(train_ds)))
                    meta_info.append('===> loss:{:.4g}'.format(loss_record))
                    if monitors_record:
                        for k,v in monitors_record.items(): 
                            meta_info.append(f'{k}:{v:.4g}')

                    loss_record, monitors_record = 0, {}
                    log_output(", ".join(meta_info))
                    torch.cuda.empty_cache()
            
            time_iter_start = time.time()
            clock.tick()
        
        clock.tock()
        try:
            # save check point
            if RANK == 0:
                if (clock.epoch+1) %  config.checkpoint_interval == 0:
                    sess.save_checkpoint('epoch-{}'.format(clock.epoch))
                sess.save_checkpoint('latest')
        except Exception:
            traceback.print_exc()
            exit(1)

    logger.info("Training is done, exit.")
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exit.")
        os._exit(0)
