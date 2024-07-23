import torch
import torch.nn as nn
import numpy as np
import os

class Utils():

    def initialize_model(model, initialization_method, nonlinearity):

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if initialization_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode ='fan_in', nonlinearity = nonlinearity)
                elif initialization_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode ='fan_in', nonlinearity = nonlinearity)
                elif initialization_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
                elif initialization_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def enforce_constraint(model):
        with torch.no_grad():
            model.weights /= torch.sum(model.weights)

    def select_activation(nonlinearity, leaky_relu_slope = 0.01):
          if nonlinearity == 'relu':
              activation = nn.ReLU()
          elif nonlinearity == 'leaky_relu':
              activation = nn.LeakyReLU(leaky_relu_slope)
          elif nonlinearity == 'selu':
              activation = nn.SELU()
          elif nonlinearity == 'elu':
              activation = nn.ELU()
          elif nonlinearity == 'silu':
              activation = nn.SiLU()
          elif nonlinearity == 'hard_swish':
              activation = nn.Hardswish()
          elif nonlinearity == 'tanh':
              activation = nn.Tanh()

          return activation

    def generate_hyperparameters(depth, layer_type, hidden_channels, NX, NY, seed, hidden_channel_factor, batch_size, n_gpus):

        input_kernel_size = [[] for i in range(depth)]
        input_stride = [[] for i in range(depth)]
        input_padding = [[] for i in range(depth)]
        input_dilation = [[] for i in range(depth)]
        input_num_layers = [[0,0,0] for i in range(depth)]
        initial_states = [[] for i in range(depth)]

        for i in range(depth):
            for j in layer_type[i]:

                if j == 'CR' or j == 'CRD' or j == 'MP' or j == 'AP':
                    input_num_layers[i][0] += 1
                elif j == 'LSTM' or j == 'LSTM_F':
                    input_num_layers[i][1] += 1
                elif j == 'CTR' or j == 'CTRU' or j == 'UP':
                    input_num_layers[i][2] += 1

                if j == 'CR' or j == 'CTR' or j == 'UP' or j == 'LSTM' or j == 'LSTM_F':
                    input_kernel_size[i].append(3)
                    input_stride[i].append(1)
                    input_padding[i].append(1)
                    input_dilation[i].append(1)
                elif j == 'CRD':
                    input_kernel_size[i].append(3)
                    input_stride[i].append(2)
                    input_padding[i].append(1)
                    input_dilation[i].append(1)
                elif j == 'MP' or j == 'AP':
                    input_kernel_size[i].append(2)
                    input_stride[i].append(2)
                    input_padding[i].append(1)
                    input_dilation[i].append(1)
                elif j == 'CTRU':
                    input_kernel_size[i].append(4)
                    input_stride[i].append(2)
                    input_padding[i].append(1)
                    input_dilation[i].append(1)

                hidden_channel_min = max(hidden_channels[0])

                for n in range(input_num_layers[i][1]):

                    initial_states[i].append((torch.randn(int(batch_size/n_gpus), int(hidden_channel_min*(hidden_channel_factor**i)), int(NX/(2**i)), int(NY/(2**i))),
                                              torch.randn(int(batch_size/n_gpus), int(hidden_channel_min*(hidden_channel_factor**i)), int(NX/(2**i)), int(NY/(2**i))),
                                              torch.zeros(int(batch_size/n_gpus), int(hidden_channel_min*(hidden_channel_factor**i)), int(NX/(2**i)), int(NY/(2**i)))))

        return input_kernel_size, input_stride, input_padding, input_dilation, input_num_layers, initial_states

    def save_checkpoint(model, optimizer, scheduler, loss_weights, save_dir):

        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss_weights' : loss_weights}, save_dir)


    def load_checkpoint(model, optimizer, scheduler, save_dir):

        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Pretrained model loaded!')

        return model, optimizer, scheduler

    def load_loss_weights(save_dir):
        checkpoint = torch.load(save_dir)
        loss_weights = checkpoint['loss_weights']
        return loss_weights

    def load_checkpoint_model(model, save_dir):
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def summary_parameters(model):
        for i in model.parameters():
            print(i.shape)

    def frobenius_norm(tensor):
        return np.sqrt(np.sum(tensor ** 2))

    def create_project(result_path, run_name):

        path_fig = result_path + run_name + '/figures/'
        path_model = result_path + run_name + '/models/'
        path_output = result_path + run_name + '/outputs/'
        path_stats = result_path + run_name + '/stats/'

        if os.path.exists(result_path) == False:
            os.mkdir(result_path)

        if os.path.exists(result_path + run_name + '/') == False:
            os.mkdir(result_path + run_name + '/')
            
        if os.path.exists(path_fig) == False:
            os.mkdir(path_fig)
        if os.path.exists(path_model) == False:
            os.mkdir(path_model)
        if os.path.exists(path_output) == False:
            os.mkdir(path_output)
        if os.path.exists(path_stats) == False:
            os.mkdir(path_stats)
        
        return path_fig, path_model, path_output, path_stats

    def dict_to_txt(base_path, result_name, run_name, param_dict, inference):
              
        indent=""
        result_path = base_path + 'results/' + result_name + '/'
        path_parameters = result_path + run_name + '/parameters/'
        
        if os.path.exists(result_path) == False:
            os.mkdir(result_path)
            
        if os.path.exists(result_path + run_name + '/') == False:
            os.mkdir(result_path + run_name + '/')

        if os.path.exists(path_parameters) == False:
            os.mkdir(path_parameters)

        if inference == True:
            f = open(path_parameters + 'inference_params.txt', "w")
        else:
            f = open(path_parameters + 'params.txt', "w")
        f.write("\n")

        for key, value in param_dict.items():
            f.write(f"{indent}{key}: ")

            if isinstance(value, list):
                i = 0
                first_item = True
                for sublist in value:
                    if first_item:
                        first_item = False
                    else:
                        f.write(" " * len(key) + "  ")

                    if isinstance(sublist, list):
                        f.write(f"{sublist}\n")
                        i += 1

                        if i == len(value):
                            f.write("\n")

                    else:
                        f.write(f"{sublist}\n")
            else:
                f.write(f"{value}\n\n")

    def generate_learning_rates(total_iterations, initial_lr, gamma, step):
        learning_rates = []
        lr = initial_lr

        for iteration in range(total_iterations):
            learning_rates.append(lr)
            if (iteration + 1) % step == 0:
                lr *= gamma

        return learning_rates

    def calculate_errors(output, true):

        error = Utils.frobenius_norm(true-output) / Utils.frobenius_norm(true)
        error_r = Utils.frobenius_norm(true[:,:,0,:,:] - output[:,:,0,:,:]) / Utils.frobenius_norm(true[:,:,0,:,:])
        error_u = Utils.frobenius_norm(true[:,:,1,:,:] - output[:,:,1,:,:]) / Utils.frobenius_norm(true[:,:,1,:,:])
        error_v = Utils.frobenius_norm(true[:,:,2,:,:] - output[:,:,2,:,:]) / Utils.frobenius_norm(true[:,:,2,:,:])
        error_E = Utils.frobenius_norm(true[:,:,3,:,:] - output[:,:,3,:,:]) / Utils.frobenius_norm(true[:,:,3,:,:])

        return error, error_r, error_u, error_v, error_E

    def calculate_errors_seq(output, true):

        size = np.size(true, 1)
        error_seq = [Utils.frobenius_norm(true[:,i,:,:,:]-output[:,i,:,:,:]) / Utils.frobenius_norm(true[:,i,:,:,:]) for i in range(size)]
        error_r_seq = [Utils.frobenius_norm(true[:,i,0,:,:]-output[:,i,0,:,:]) / Utils.frobenius_norm(true[:,i,0,:,:]) for i in range(size)]
        error_u_seq = [Utils.frobenius_norm(true[:,i,1,:,:]-output[:,i,1,:,:]) / Utils.frobenius_norm(true[:,i,1,:,:]) for i in range(size)]
        error_v_seq = [Utils.frobenius_norm(true[:,i,2,:,:]-output[:,i,2,:,:]) / Utils.frobenius_norm(true[:,i,2,:,:]) for i in range(size)]
        error_E_seq = [Utils.frobenius_norm(true[:,i,3,:,:]-output[:,i,3,:,:]) / Utils.frobenius_norm(true[:,i,3,:,:]) for i in range(size)]

        return error_seq, error_r_seq, error_u_seq, error_v_seq, error_E_seq