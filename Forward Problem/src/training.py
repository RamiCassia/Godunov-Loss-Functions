import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

from src.utilities import Utils
from src.plotting import Plots
from src.loss import loss_generator

import numpy as np
import pickle
import random
from tqdm import tqdm
import time as tm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Train():

    @torch.jit._script_if_tracing
    def train(model, train_loader, initial_states, n_iters, time_steps, time_batch_size, learning_rate,
          t0_index, dt, dx, dy, pre_run_name, num_time_batch, scheduler_step, scheduler_gamma, reset_lr,
          base_epoch, print_interval, NX, NY, gamma, run_name, n_cols, depth, loss_type, adaptive_loss, adaptive_epsilon, weighted_loss, channel_wise_backprop, base_path, result_name, train_range_list, device, e_visc, e_tvd, e_ent):

        train_loss_list, train_loss_list_r, train_loss_list_u, train_loss_list_v, train_loss_list_E = [], [], [], [], []
        train_error_list, train_error_list_r, train_error_list_u, train_error_list_v, train_error_list_E = [], [], [], [], []
        loss_weight_r, loss_weight_u, loss_weight_v, loss_weight_E = [], [], [], []
        second_last_states = [[] for i in range(depth)]
        prev_output = []
        error_best = 1e10

        result_path = base_path + 'results/' + result_name + '/'

        path_fig, path_model, path_output, path_stats = Utils.create_project(result_path, run_name)

        with open(path_stats + 'train_ranges.pkl', 'wb') as file:
            pickle.dump(train_range_list, file)

        if pre_run_name == None:
            loss_weights = nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True))
        else:
            loss_weights = Utils.load_loss_weights(result_path + pre_run_name + '/models/model.pt')
            loss_weights.requires_grad = True

        parameters_CNN = [{"params": model.parameters(), "lr": learning_rate}]
        parameters_Loss = [{"params": loss_weights, "lr": learning_rate}]

        if weighted_loss == True:
            optimizer = optim.Adam(parameters_CNN + parameters_Loss)
        else:
            optimizer = optim.Adam(parameters_CNN)

        scheduler = StepLR(optimizer, step_size = scheduler_step, gamma = scheduler_gamma)

        if pre_run_name == None:
            pre_model_save_path = result_path + run_name + '/models/checkpointInit.pt'
            Utils.save_checkpoint(model, optimizer, scheduler, loss_weights, pre_model_save_path)
        else:
            pre_model_save_path = result_path + pre_run_name + '/models/model.pt'

        model, optimizer, scheduler = Utils.load_checkpoint(model, optimizer, scheduler, pre_model_save_path)

        loss_func = loss_generator(dt = dt, dx = dx, dy = dy, gamma = gamma, NX = NX, NY = NY, loss_type = loss_type, loss_weights = loss_weights, channel_wise_backprop = channel_wise_backprop, device = device, e_visc = e_visc, e_tvd = e_tvd, e_ent = e_ent, adaptive = adaptive_loss, epsilon = adaptive_epsilon)

        for param_group in optimizer.param_groups:
            if reset_lr == True:
                param_group['lr'] = learning_rate
            print(param_group['lr'])

        learning_rates = Utils.generate_learning_rates(n_iters, param_group['lr'], scheduler_gamma, scheduler_step)


        if pre_run_name is not None:
            with open(result_path + pre_run_name + '/stats/' + 'losses.pkl', 'rb') as file:
                prev_loss_dict = pickle.load(file)
            with open(result_path + pre_run_name + '/stats/' + 'errors.pkl', 'rb') as file:
                prev_error_dict = pickle.load(file)
            with open(result_path + pre_run_name + '/stats/' + 'loss_weights.pkl', 'rb') as file:
                prev_loss_weight_dict = pickle.load(file)

            base_epoch = None

            for key, value in prev_loss_dict.items():
                if isinstance(value, list) and len(value) > 0:
                    base_epoch = len(value)
                    break

            prev_lr = np.load(result_path + pre_run_name + '/stats/' + 'lr.npy')
            learning_rates = np.concatenate((prev_lr[0 : base_epoch], learning_rates), axis=0)
        else:
            base_epoch = 0

        np.save(path_stats + 'lr' + '.npy', learning_rates)

        for epoch in tqdm(range(base_epoch, base_epoch + n_iters)):

            output_list, target_list = [], []
            epoch_loss, epoch_loss_r, epoch_loss_u, epoch_loss_v, epoch_loss_E = 0, 0, 0, 0, 0
            batch_epoch_loss, batch_epoch_loss_r, batch_epoch_loss_u, batch_epoch_loss_v, batch_epoch_loss_E = 0, 0, 0, 0, 0

            idx = 0

            for idx, (features, targets) in enumerate(train_loader):

                time_batch_loss, time_batch_loss_r, time_batch_loss_u, time_batch_loss_v, time_batch_loss_E = 0, 0, 0, 0, 0

                for time_batch_id in range(num_time_batch):
                    if time_batch_id == 0:
                        hidden_states = initial_states
                        u0 = features[:]
                    else:
                        hidden_states = states_detached
                        u0 = prev_output[:,-2:-1,:,:,:].detach()

                    output, second_last_states = model(u0)
                    output = torch.cat((u0, output), dim=1)

                    if channel_wise_backprop == True:
                        loss, loss_r, loss_u, loss_v, loss_E = loss_func(output)
                        all_losses = [loss_r, loss_u, loss_v, loss_E]
                        random.shuffle(all_losses)
                        n = 0
                        for loss in all_losses:
                            retain_graph = True if n != (len(all_losses) - 1) else False
                            optimizer.zero_grad()
                            loss.backward(retain_graph = retain_graph)
                            optimizer.step()
                            n += 1
                        scheduler.step()
                    else:
                        loss, loss_r, loss_u, loss_v, loss_E = loss_func(output)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        Utils.enforce_constraint(loss_func)

                    time_batch_loss += loss.item()
                    time_batch_loss_r += loss_r.item()
                    time_batch_loss_u += loss_u.item()
                    time_batch_loss_v += loss_v.item()
                    time_batch_loss_E += loss_E.item()

                    batch_epoch_loss += time_batch_loss
                    batch_epoch_loss_r += time_batch_loss_r
                    batch_epoch_loss_u += time_batch_loss_u
                    batch_epoch_loss_v += time_batch_loss_v
                    batch_epoch_loss_E += time_batch_loss_E

                    prev_output = output
                    states_detached = [[] for i in range(depth)]
                    for i in range(len(second_last_states)):
                        state_list = second_last_states[i]
                        for s in state_list:
                            states_detached[i].append((s[0].detach(), s[1].detach(), s[2].detach()))

                    if epoch % (print_interval) == 0:
                        output_list_append = [output[i, :, :, :, :].reshape(-1,4,NX,NY).cpu().detach().numpy() for i in range(output.size(0))]
                        target_list_append = [targets[i, :, :, :, :].reshape(-1,4,NX,NY).cpu().detach().numpy() for i in range(targets.size(0))]
                        output_list = output_list + output_list_append
                        target_list = target_list + target_list_append
                        del output_list_append, target_list_append

                batch_epoch_loss, batch_epoch_loss_r, batch_epoch_loss_u, batch_epoch_loss_v, batch_epoch_loss_E = batch_epoch_loss/(time_batch_id + 1), batch_epoch_loss_r/(time_batch_id + 1), batch_epoch_loss_u/(time_batch_id + 1), batch_epoch_loss_v/(time_batch_id + 1), batch_epoch_loss_E/(time_batch_id + 1)

                epoch_loss += batch_epoch_loss
                epoch_loss_r += batch_epoch_loss_r
                epoch_loss_u += batch_epoch_loss_u
                epoch_loss_v += batch_epoch_loss_v
                epoch_loss_E += batch_epoch_loss_E

            epoch_loss, epoch_loss_r, epoch_loss_u, epoch_loss_v, epoch_loss_E = epoch_loss/(idx+1), epoch_loss_r/(idx+1), epoch_loss_u/(idx+1), epoch_loss_v/(idx+1), epoch_loss_E/(idx+1)

            output_list = np.array(output_list)

            train_loss_list.append(epoch_loss); train_loss_list_r.append(epoch_loss_r); train_loss_list_u.append(epoch_loss_u); train_loss_list_v.append(epoch_loss_v); train_loss_list_E.append(epoch_loss_E)
            loss_weight_r.append(loss_func.weights[0].item()); loss_weight_u.append(loss_func.weights[1].item()); loss_weight_v.append(loss_func.weights[2].item()); loss_weight_E.append(loss_func.weights[3].item())


            if (epoch) % (print_interval) == 0:

                error, error_r, error_u, error_v, error_E = Plots.post_process_train(np.array(output_list), np.array(target_list), fig_save_path = path_fig, epoch = epoch, loss = epoch_loss, n_cols = n_cols)

                if error < error_best:
                    error_best = error
                Utils.save_checkpoint(model, optimizer, scheduler, loss_weights, path_model + 'model.pt')

                train_error_list.append(error); train_error_list_r.append(error_r); train_error_list_u.append(error_u); train_error_list_v.append(error_v); train_error_list_E.append(error_E)

                loss_dict = {'Overall':train_loss_list, 'Density':train_loss_list_r, 'x-Velocity':train_loss_list_u, 'y-Velocity':train_loss_list_v, 'Energy':train_loss_list_E}
                error_dict = {'Overall':train_error_list, 'Density':train_error_list_r, 'x-Velocity':train_error_list_u, 'y-Velocity':train_error_list_v, 'Energy':train_error_list_E}
                loss_weight_dict = {'Overall' : [0], 'Density': loss_weight_r, 'x-Velocity': loss_weight_u, 'y-Velocity': loss_weight_v, 'Energy': loss_weight_E}

                if pre_run_name is not None:
                    loss_dict = {key: prev_loss_dict[key] + loss_dict[key] for key in prev_loss_dict}
                    error_dict = {key: prev_error_dict[key] + error_dict[key] for key in prev_error_dict}
                    loss_weight_dict = {key: prev_loss_weight_dict[key] + loss_weight_dict[key] for key in prev_loss_weight_dict}

                with open(path_stats + 'losses.pkl', 'wb') as file:
                    pickle.dump(loss_dict, file)
                with open(path_stats + 'errors.pkl', 'wb') as file:
                    pickle.dump(error_dict, file)
                with open(path_stats + 'loss_weights.pkl', 'wb') as file:
                    pickle.dump(loss_weight_dict, file)

                Plots.plot_dict_list(loss_dict, learning_rates[0:epoch], title="Loss Evolution", x_label="Epoch", y_label="Loss", log_y=True, color_background=True, include = [0,1,2,3,4], print_interval = 1, fig_save_path = path_fig, fig_name = 'loss')
                Plots.plot_dict_list(error_dict, learning_rates[0:epoch], title="Error Evolution", x_label="Epoch", y_label="L2 Norm", log_y=False, color_background=True, include = [0,1,2,3,4], print_interval = print_interval, fig_save_path = path_fig, fig_name = 'error')

                if weighted_loss == True:
                    Plots.plot_dict_list(loss_weight_dict, learning_rates[0:epoch], title="Loss Weight Evolution", x_label="Epoch", y_label="Weight", log_y=False, color_background=True, include = [1,2,3,4], print_interval = 1, fig_save_path = path_fig, fig_name = 'loss_weights', leg_loc = 'center right')


            print("\r", end = '[%d/%d %d%%] loss: %.10f loss_r: %.10f loss_u: %.10f loss_v: %.10f loss_E: %.10f' % ((epoch), (base_epoch + n_iters - 1), ((epoch)/(base_epoch + n_iters)*100.0), epoch_loss, epoch_loss_r, epoch_loss_u, epoch_loss_v, epoch_loss_E), flush=True)

        np.save(path_output + '.npy', output_list)