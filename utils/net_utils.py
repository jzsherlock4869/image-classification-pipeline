import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

def get_bare_model(net):
    if isinstance(net, DistributedDataParallel) or isinstance(net, DataParallel):
        net = net.module
    return net

def load_network(net, load_path):
    state_dict = torch.load(
        load_path, map_location=lambda storage, loc: storage
    )#['params']
    net = get_bare_model(net)
    net.load_state_dict(state_dict)
    return net