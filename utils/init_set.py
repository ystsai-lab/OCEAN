import numpy as np
import torch
# coding=utf-8
from matplotlib import pyplot as plt
from utils.prototypical_batch_sampler import PrototypicalBatchSampler

def init_sampler(labels, way, support, query, iterations):
    classes_per_it = way
    num_samples = support + query

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)


def init_seed(seed=1234):
    '''
        Initialize random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_optim( model, learning_rate=0.001):
    '''
        Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def init_lr_scheduler( optim, lr_scheduler_gamma=0.1, lr_scheduler_step=20):
    '''
        Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=lr_scheduler_gamma,
                                           step_size=lr_scheduler_step)
