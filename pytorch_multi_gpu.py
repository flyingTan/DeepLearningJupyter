import torch
import argparse
import os
import  torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

# distribute package
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils import get_logger

## 1. Model defination
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 10)
    
    def forward(self, x):
        return F.relu(self.net2(F.relu(self.net1(x))))

## 2. Dataset setting
class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(128, 1, 10)
        self.label = self.data ** 2
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        assert index < self.data.shape[0]

        return self.data[index], self.label[index]

def setup(rank, seed = -1):
    dist.init_process_group(backend = "nccl",
                            init_method= "env://")
    torch.cuda.set_device(rank)
    if seed > 0 :
        torch.manual_seed(seed)

def cleanup():
    dist.destroy_process_group()

def training(rank):
    if rank == 0:
        global logger
        logger = get_logger(__name__, "train.log")
    setup(rank)

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas = dist.get_world_size(), rank= rank)
    data_loader = DataLoader(dataset, 
                             batch_size = 2, 
                             shuffle= False, 
                             sampler = sampler)
     
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 1e-4)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids= [rank], output_device = rank)
    
    loss_fn = nn.MSELoss()
    epochs = 200

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for ite, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if dist.get_rank() == 0:
            logger.info("pid:%s rank:%s epoch:%s loss:%s batch_size:%s"%(os.getpid(), rank, epoch, loss.item(), inputs.shape[0]))

        if epoch == epochs - 1 and dist.get_rank() == 0:
            torch.save(model.state_dict(), "toy.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
 
    args = parser.parse_args()
    rank = args.local_rank
    training(rank)
