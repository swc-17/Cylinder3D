# -*- coding:utf-8 -*-
# author: swc
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from utils.log_util import Logger

import warnings
import ipdb
warnings.filterwarnings("ignore")


def main(rank, args):	 
    args.rank = rank
    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    # distributed training 
    torch.cuda.set_device(rank)
    my_model.cuda(rank)

    if args.gpus > 1:                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=args.gpus,                              
            rank=rank                                               
        )   
        my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank])

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
                                            
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  args=args,
                                                                )
    # training
    logger = Logger('exp', rank)
    os.system(f'cp {args.config_path} {logger.log_dir}')
    model_save_path = logger.log_dir

    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        logger.write('epoch: {} |'.format(epoch))

        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0 and epoch >= 0:
                print('start validating')
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda(rank) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).cuda(rank) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).cuda(rank)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                logger.write('Validation per class iou: \n')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%\n' % (class_name, class_iou * 100))
                    logger.write('%s : %.2f%%\n' % (class_name, class_iou * 100))
                    logger.scalar_summary('val_{class_name}_iou', class_iou * 100, global_iter)
                val_miou = np.nanmean(iou) * 100
                logger.scalar_summary('val_miou', val_miou, global_iter)
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))
                logger.write('Current val miou is %.3f while the best val miou is %.3f\n' %
                      (val_miou, best_val_miou))
                logger.write('Current val loss is %.3f\n' %
                      (np.mean(val_loss_list)))                      

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda(rank) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).cuda(rank) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).cuda(rank)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            loss0 = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0)
            loss1 = loss_func(outputs, point_label_tensor)
            loss =  loss0 + loss1 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                    logger.write('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                    logger.scalar_summary('train_loss', loss, global_iter)
                    logger.scalar_summary('train_loss_lovasz', loss0, global_iter)
                    logger.scalar_summary('train_loss_ce', loss1, global_iter)
                else:
                    print('loss error')

            optimizer.zero_grad()
            if rank == 0:
                pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                    logger.write('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                    logger.scalar_summary('train_loss', loss, global_iter)
                    logger.scalar_summary('train_loss_lovasz', loss0, global_iter)
                    logger.scalar_summary('train_loss_ce', loss1, global_iter)
                else:
                    print('loss error')
        pbar.close()
        torch.save(my_model.state_dict(), f'{model_save_path}/{epoch}.pth')
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', default='waymo', type=str)
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus')
    args = parser.parse_args()

    assert args.data in ['semantickitti', 'nuScenes', 'waymo'], 'Dataset not supported.'
    args.config_path = f'config/{args.data}.yaml'
    args.world_size = args.gpus

    if args.gpus > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'              
        os.environ['MASTER_PORT'] = '9999' 
        mp.spawn(main, nprocs=args.gpus, args=(args,))         
    else:
        main(0, args)
