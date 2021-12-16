import os
from pathlib import  Path
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tutils
from torchvision import transforms as trans

from data.ms1m import get_train_loader
from data.lfw import LFW

from backbone.arcfacenet import SEResNet_IR
from margin.ArcMarginProduct import ArcMarginProduct

from util.utils import save_checkpoint, test


"""Configuration"""
conf = edict()

conf.train_root = "dataset/MS1M"
conf.lfw_test_root = "dataset/lfw_aligned_112"
conf.lfw_file_list = "dataset/lfw_pair.txt"

conf.mode = "se_ir"
conf.depth = 100
conf.margin_type = "ArcFace"
conf.feature_dim = 512       #Özellik Boyutu
conf.scale_size0  = 32
conf.batch_size = 96
conf.learning_rate = 0.01
conf.milestone = [8,10,12]
conf.total_epoch = 14

conf.save_folder = "dataset/saved"
conf.save_dir = os.path.join(conf.save_folder, conf.mode + "_" +str(conf.depth))
conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conf.num_workers = 4
conf.pin_memory = True

os.makedirs(conf.save_dir, exist_ok = True)


"""Data Loading"""
transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))
])

trainloader, class_num = get_train_loader(conf)

print("number of id:",class_num)
print(trainloader.dataset)

lfw_dataset = LFW(conf.lfw_test_root, conf.lfw_file_list, transform = transform)
lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size = 128, num_workers =conf.num_workers)

"""Creating Model"""

net = SEResNet_IR(conf.depth, feature_dim= conf.feature_dim, mode= conf.mode).to(conf.device)

print(conf.device)