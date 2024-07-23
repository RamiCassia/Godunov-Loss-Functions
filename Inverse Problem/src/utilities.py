import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Utils():

    def initialize_model(model, initialization_method, nonlinearity):

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if initialization_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode ='fan_in', nonlinearity = nonlinearity)
                elif initialization_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode ='fan_in', nonlinearity = nonlinearity)
                elif initialization_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
                elif initialization_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
                elif initialization_method == 'constant':
                    nn.init.constant_(m.weight, 0.01)
                elif initialization_method == 'uniform_dist':
                    nn.init.uniform_(m.weight, a=0.005, b=0.01)
                elif initialization_method == 'dirac':
                    nn.init.dirac_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def save_model_opt(model, optimizer, path):
        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, path + 'model_and_optimizer.pth')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def summary_parameters(model):
        for i in model.parameters():
            print(i.shape)

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


    def dict_to_txt(path, param_dict):

        indent=""

        f = open(path + 'params.txt', "w")
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


    def interpolate(x, scale_factor, mode):

        a,b,c,d,e = x.shape

        x = x.reshape(a*b, c, d, e)

        x = F.interpolate(x, scale_factor= scale_factor, mode=mode, align_corners=False)

        x = x.reshape(a,b, c, d*scale_factor, e*scale_factor)

        return x