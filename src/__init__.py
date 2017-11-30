# encoding: utf-8

import os
# Configure visible GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# Import self-define libraries
from . import Dataset
from . import AttentionModel
from . import Experiments
from . import SuperClass
from . import tools

Proj_HomeDir = '/home/jack/Workspace/Python/RADCN'
