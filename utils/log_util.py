# -*- coding:utf-8 -*-
# author:swc
# @file: log_util.py 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
  import tensorboardX
  # print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, save_dir, rank):
    """Create a summary writer logging to log_dir."""
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = save_dir + '/logs_{}'.format(time_str)
    self.log_dir = log_dir
    self.rank = rank
    if self.rank != 0:
        return
    os.mkdir(log_dir)

    file_name = os.path.join(log_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')


    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    self.log = open(log_dir + '/log.txt', 'w')
    self.start_line = True

  def write(self, txt):
    if self.rank != 0:
        return

    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    if self.rank != 0:
        return

    self.log.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if self.rank != 0:
        return

    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)
