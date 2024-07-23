import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import numpy as np
import pickle as pkl
import torch
import torch.optim as optim
import psutil
from src.utilities import Utils
from src.data_handling import Dataloader
from src.model import Super
from src.loss import loss_generator
from src.plotting import Plots
import random
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--con", type = int)
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

    result_path = base_path + 'results/'
    run_name = 'run_name'

    path_fig, path_model, path_output, path_stats = Utils.create_project(result_path, run_name)

    file_name_conv = 'WENO_2x4x256x256_'

    input_channels = 4
    hidden_channels = 256
    seed = 85
    time_steps = 1
    n_time_steps = 1
    NX = 256
    NY = 256
    dt = 0.00005
    dx = 0.4/(NX-1)
    dy = 0.4/(NY-1)
    gamma = 1.4
    t0_index = 0
    configurations = [inputs.con]
    scale_factor = 16

    batch_size = 6
    data_split = 1.0
    shuffle_data = False
    spacing = 0
    pool_mode = 'avg'
    interp_mode = 'bilinear'
    loss_type = inputs.loss_type
    e_tvd = inputs.visc
    e_ent = inputs.ent
    e_visc = inputs.tvd
    train_loader, _, _ = Dataloader(configurations, NX, NY, t0_index, time_steps, n_time_steps, base_path, file_name_conv, pool_mode, scale_factor, data_split, batch_size, seed, shuffle_data, device, spacing, True).generate_loader()
    plot_interval = 100
    initialization = 'xavier_normal'
    learning_rate = 0.00005
    reg_weight = 25

    model = Super(in_channels = input_channels, hidden_channels = hidden_channels, scale_factor = scale_factor, mode = interp_mode).cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters in the model:", num_params)
    Utils.initialize_model(model, initialization, 'relu')
    loss_func = loss_generator(dt = dt, dx = dx, dy = dy, gamma = gamma, NX = NX, NY = NY, loss_type = loss_type, device = device, e_tvd = e_tvd, e_ent = e_ent, e_visc = e_visc, scale_factor = scale_factor, pool_mode = pool_mode)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = inputs.nadamopt

    error_bilinear_l = []
    error_bicubic_l = []

    param_dict = {'Run Name':run_name,'dt':dt, 'NX': NX, 'NY':NY, 'Initial Index' : t0_index, 'Input Channels' : input_channels,
                'Hidden Channels' : hidden_channels ,'Random Seed' : seed, 'Configurations': tuple(configurations), 'Interpolator' : interp_mode, 'Pooling' : pool_mode,
                'Learning Rate' : learning_rate, 'Initialization' : initialization, 'Scale Factor' : scale_factor, 'Epochs' : num_epochs, 'e_tvd' : e_tvd, 'e_ent' : e_ent,
                'Gamma' : gamma, 'Loss Type' : loss_type, 'Reg Weight' : reg_weight}

    Utils.dict_to_txt(path_stats, param_dict)

    for inputs, targets in train_loader:
        bilinear_a = Utils.interpolate(inputs, scale_factor, 'bilinear')
        bicubic_a = Utils.interpolate(inputs, scale_factor, 'bicubic')

        for i in range(len(configurations)):

            error_bilinear = torch.norm(bilinear_a[i] - targets[i])/torch.norm(targets[i])
            error_bicubic = torch.norm(bicubic_a[i] - targets[i])/torch.norm(targets[i])

            error_bilinear_l.append(error_bilinear.cpu().detach().numpy())
            error_bicubic_l.append(error_bicubic.cpu().detach().numpy())

        np.save(path_output + 'inputs' + '.npy', inputs.cpu().detach().numpy())
        np.save(path_output + 'targets' + '.npy', targets.cpu().detach().numpy())
        np.save(path_output + 'bilinear' + '.npy', bilinear_a.cpu().detach().numpy())
        np.save(path_output + 'bicubic' + '.npy', bicubic_a.cpu().detach().numpy())


    def Train():

        loss_list = []
        reg_loss_list = []
        error_list = []
        error_rho_list = []

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, reg_loss = loss_func(inputs, outputs, bicubic_a, targets, reg_weight)
                loss.backward()
                optimizer.step()
                phy_loss = loss - reg_loss

                print("\r", end = '[%d/%d %d%%] loss: %.10f reg_loss: %.10f phy_loss: %.10f' % ((epoch), (num_epochs - 1), ((epoch)/(num_epochs)*100.0), loss.item(), reg_loss.item(), phy_loss.item()))

                loss_list.append(loss.cpu().detach().numpy())
                reg_loss_list.append(reg_loss.cpu().detach().numpy())

                if epoch % plot_interval == 0 and epoch != 0:

                    error_l = []
                    error_rho_l = []
                    for i in range(len(configurations)):
                        error = np.linalg.norm(targets[i,-1].cpu().detach().numpy()- outputs[i,-1].cpu().detach().numpy())/np.linalg.norm(targets[i,-1].cpu().detach().numpy())
                        error_l.append(error)

                        error_rho = np.linalg.norm(targets[i,-1,0].cpu().detach().numpy()- outputs[i,-1,0].cpu().detach().numpy())/np.linalg.norm(targets[i,-1,0].cpu().detach().numpy())
                        error_rho_l.append(error_rho)

                    error_list.append(error_l)
                    error_rho_list.append(error_rho_l)

                    Plots.plot_outputs(inputs.cpu().detach().numpy(), outputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), bilinear_a.cpu().detach().numpy(), bicubic_a.cpu().detach().numpy(), error_l, error_bilinear_l, error_bicubic_l, epoch, path_fig,0)

                    np.save(path_output + 'outputs_' + str(epoch) + '.npy', outputs.cpu().detach().numpy())

                    with open(path_stats + 'loss_list.pkl', 'wb') as f:
                        pkl.dump(loss_list, f)
                    with open(path_stats + 'reg_loss_list.pkl', 'wb') as f:
                        pkl.dump(reg_loss_list, f)
                    with open(path_stats + 'error_list.pkl', 'wb') as f:
                        pkl.dump(error_list, f)
                    with open(path_stats + 'error_rho_list.pkl', 'wb') as f:
                        pkl.dump(error_rho_list, f)

        with open(path_stats + 'loss_list.pkl', 'wb') as f:
            pkl.dump(loss_list, f)
        with open(path_stats + 'reg_loss_list.pkl', 'wb') as f:
            pkl.dump(reg_loss_list, f)
        with open(path_stats + 'error_list.pkl', 'wb') as f:
            pkl.dump(error_list, f)
        with open(path_stats + 'error_rho_list.pkl', 'wb') as f:
            pkl.dump(error_rho_list, f)

        Utils.save_model_opt(model, optimizer, path_model)
        return True

    _ = Train()

if __name__ == '__main__':
    main()