
import psutil
import torch


num_cpus = psutil.cpu_count(logical=False) - 2
cpu_device = torch.device("cpu")

num_gpus = \
    len([torch.cuda.device(g) for g in range(torch.cuda.device_count())])
gpu_device = torch.device("cuda")

class_device = cpu_device
learning_device = gpu_device
