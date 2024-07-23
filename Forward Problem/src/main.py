import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

from src.utilities import Utils
from src.data_handling import Dataloader
from src.model import PhyCR_UNet
from src.training import Train

import random
import gc
import torch
import numpy as np
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--con", type = int)
    parser.add_argument("--tp", type = int)
    parser.add_argument("--tin", type = int)
    parser.add_argument("--nadamopt", type = int)
    parser.add_argument("--seed", type = int)
    parser.add_argument("--loss_type", type = str)
    parser.add_argument("--visc", type = float)
    parser.add_argument("--ent", type = float)
    parser.add_argument("--tvd", type = float)
    args=parser.parse_args()
    return args

def main():

    inputs = parse_args()

    seed = inputs.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    config = inputs.con
    tp = inputs.tp 
    result_name = 'seed_' + str(inputs.seed) + '_con_' + str(inputs.con) + '_loss_' + inputs.loss_type

    if inputs.loss_type == 'PINN':
        result_name = result_name + '_visc_' + str(inputs.visc) + '_tvd_' + str(inputs.tvd) + '_ent_' + str(inputs.ent)

    for t in range(inputs.tin,80,5): 

        model, train_loader, test_loader, initial_states, output = None, None, None, None, None
        del model, train_loader, test_loader, initial_states, output
        torch.cuda.empty_cache()
        gc.collect()

        run_name = 'c'+ str(config) + '_t' + str(t)
        pre_run_name = None if tp == 0 else 'c'+ str(config) + '_t' + str(tp) 
        file_name_conv = 'WENO_151x4x128x128_'

        seed = 85
        time_steps = t
        n_time_steps = 1
        NX = 128
        NY = 128
        dt = 0.002
        dx = 1/(NX-1)
        dy = 1/(NY-1)
        gamma = 1.4
        t0_index = 0 
        configurations = [config]
        batch_size = 1
        data_split = 1.0
        shuffle_data = True
        spacing = 0
        attention = None
        initialization = 'kaiming_uniform'
        time_batch_size = time_steps - 1
        steps = time_batch_size + 1
        num_time_batch = int(time_steps / time_batch_size)
        base_epoch = 0

        lr_adam = 3e-5
        n_iters_adam = inputs.nadamopt

        scheduler_step = 250
        scheduler_gamma = 1.0
        reset_lr = True
        print_interval = 1000
        n_cols = n_time_steps*len(configurations) if n_time_steps*len(configurations) <= 10 else 10
        nonlinearity = 'hard_swish'
        activation = Utils.select_activation(nonlinearity, 0.05)
        loss_type = inputs.loss_type #here
        adaptive_loss = False
        adaptive_epsilon = 0.01
        weighted_loss = False
        channel_wise_backprop = False
        e_visc = inputs.visc
        e_tvd = inputs.tvd
        e_ent = inputs.ent
        depth = 1 
        input_channels = 4
        padding_mode = 'replicate'
        effective_step = list(range(0, steps))
        hidden_channel_factor = 1

        hidden_channels = [[128,128,4]]
        layer_type = [['CR','LSTM_F','CTR']]
        batch_norm = [[False, False, False]]

        input_kernel_size, input_stride, input_padding, input_dilation, num_layers, initial_states = Utils.generate_hyperparameters(depth, layer_type, hidden_channels, NX, NY, seed, hidden_channel_factor, batch_size, n_gpus)

        param_dict = {'Run Name':run_name, 'Pre Run Name' : pre_run_name, 'Depth': depth, 'Time Steps': time_steps,'Multiples of Time Steps' : n_time_steps, 'dt':dt, 'NX': NX, 'NY':NY,
                    'Initial Index' : t0_index, 'Time Spacing': spacing, 'Input Channels' : input_channels, 'Layer Type': layer_type, 'Hidden Channels' : hidden_channels,
                    'Kernel Size' : input_kernel_size, 'Input Stride' : input_stride, 'Input Padding' : input_padding, 'Input Dilation' : input_dilation ,
                    'Num Layers' : num_layers, 'Padding Mode' : padding_mode ,'Adam Iterations' : n_iters_adam, 'Learning Rate' : lr_adam,
                    'Schedule Gamma' : scheduler_gamma, 'Schedule Step': scheduler_step, 'Random Seed' : seed,
                    'Loss Type' : loss_type, 'Adaptive Loss' : adaptive_loss, 'Adaptive Epsilon' : adaptive_epsilon,
                    'Initialization' : initialization, 'Non-linearity' : nonlinearity, 'Weighted Loss' : weighted_loss, 'Channel-wise Backprop' : channel_wise_backprop, 'Configurations': tuple(configurations), 'Attention Type' : attention, 'e_visc' : e_visc, 'e_tvd' : e_tvd, 'e_ent': e_ent}

        Utils.dict_to_txt(base_path, result_name, run_name, param_dict, inference = False)

        model = PhyCR_UNet(initial_states = initial_states,
                depth = depth,
                layer_type = layer_type,
                input_channels = input_channels,
                hidden_channels = hidden_channels,
                input_kernel_size = input_kernel_size,
                input_stride = input_stride,
                input_padding = input_padding,
                input_dilation = input_dilation,
                num_layers = num_layers,
                padding_mode = padding_mode,
                activation = activation,
                attention = attention,
                batch_norm = batch_norm,
                dt = dt,
                step = steps,
                effective_step = effective_step)

        model = model.cuda()

        Utils.initialize_model(model, initialization,'leaky_relu')

        train_loader, _, train_range_list = Dataloader(configurations, NX, NY, t0_index, time_steps, n_time_steps, base_path, file_name_conv, data_split, batch_size, seed, shuffle_data, device, spacing, True).generate_loader()
       
        Train.train(model, train_loader, initial_states, n_iters_adam, time_steps, time_batch_size,
                                    lr_adam, t0_index, dt, dx, dy, pre_run_name, num_time_batch,
                                    scheduler_step, scheduler_gamma, reset_lr, base_epoch, print_interval,
                                    NX, NY, gamma, run_name, n_cols, depth, loss_type, adaptive_loss, adaptive_epsilon, weighted_loss, channel_wise_backprop, base_path, result_name, train_range_list, device, e_visc, e_tvd, e_ent)


        tp = t


if __name__ == '__main__':
    main() 