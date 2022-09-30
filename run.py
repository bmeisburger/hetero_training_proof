#! /home/apd10/anaconda3/bin/python
import torch
torch.manual_seed(121211)
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from os import path
import os

import pdb
import argparse
from os.path import dirname, abspath, join
import glob
cur_dir = dirname(abspath(__file__))
import yaml
from research.TLoop import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                    help="config to setup the training")

results = parser.parse_args()
config_file = results.config
with open(config_file, "r") as f:
  config = yaml.load(f)
if config['module'] == "TLoop":
  run = TLoop(config)
  run.loop()
else:
  raise NotImplementedError
