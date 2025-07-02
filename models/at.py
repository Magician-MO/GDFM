import os
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import DiscPool
import util.util as util
from itertools import chain
from data import create_dataset
if __name__ == '__main__':
    model = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'dim': opt.dim,
                                         'dim_mults': opt.dim_mults,
                                         'init_dim': opt.init_dim,
                                         'resnet_block_groups': opt.groups,
                                         'learned_sinusoidal_cond': opt.learned_sinusoidal_cond,
                                         'learned_sinusoidal_dim': opt.learned_sinusoidal_dim,
                                         'random_fourier_features': opt.random_fourier_features,
                                         'time_dim_mult': opt.time_dim_mult})
    print(model)