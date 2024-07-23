import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


class Dataloader():

    def __init__(self, configurations, NX, NY, t0_index, time_steps, n_time_steps, base_path, file_name_conv, pool_mode, scale_factor, train_size, batch_size, seed, rand, device, spacing = 0, print = False):

        self.configurations = configurations
        self.NX = NX
        self.NY = NY
        self.t0_index = t0_index
        self.time_steps = time_steps
        self.n_time_steps = n_time_steps
        self.base_path = base_path
        self.file_name_conv = file_name_conv
        self.train_size = train_size
        self.batch_size = batch_size
        self.seed = seed
        self.rand = rand
        self.spacing = spacing
        self.print = print
        self.device = device
        self.pool_mode = pool_mode
        self.scale_factor = scale_factor

    def pool(self, x):

        batch, time, channel, height, width = x.shape
        x = x.reshape(-1, height, width)

        if  self.pool_mode == 'max':
            x = F.max_pool2d(x, kernel_size=(self.scale_factor, self.scale_factor))
        elif  self.pool_mode == 'avg':
            x = F.avg_pool2d(x, kernel_size=(self.scale_factor, self.scale_factor))
        elif self.pool_mode =='ss':
            x = self.dirac_conv(x)

        _,height,width = x.shape
        x = x.reshape(batch, time, channel, height, width)

        return x

    def preprocess(self):

        U = []

        data_x = []
        data_y = []
        new_truth = []
        train_range_list = []

        for i in self.configurations:
            path = self.base_path + '/data/' + self.file_name_conv
            path = path + str(i) + '.npy'

            if os.path.exists(path) == True:
                U_e = np.load(path)
                U_e = np.reshape(U_e, (-1, 4,self.NX, self.NY))
                U.append(U_e)

        U = np.array(U)
        time_steps = self.time_steps
        t0_index = self.t0_index

        for k in range(self.n_time_steps):

            spacing = 0 if k == 0 else self.spacing
            train_range = (t0_index + (time_steps + spacing)*k, t0_index + (time_steps + spacing)*k + time_steps)
            train_range_list.append(train_range)

            input = U[:, train_range[0] : train_range[1] + 1,:,:,:]
            new_truth = U[:, train_range[0] : train_range[1] + 1, :, :, :]
            data_x_e = [input[j,:,:,:,:] for j in range(np.size(U,0))]
            data_y_e = [new_truth[j,:,:,:,:] for j in range(np.size(U,0))]
            data_x = data_x + data_x_e
            data_y = data_y + data_y_e

        data_x = np.array(data_x, dtype = float)
        data_y = np.array(data_y, dtype = float)

        return data_x, data_y, train_range_list

    def split_x_y_list(self, data_x, data_y):

        if self.rand == True:
            np.random.seed(self.seed)
            data_x = np.random.permutation(data_x)

            np.random.seed(self.seed)
            data_y = np.random.permutation(data_y)

        middle_index = int(np.round(len(data_x) * self.train_size))

        data_x_train = data_x[:middle_index]
        data_x_test = data_x[middle_index:]

        data_y_train = data_y[:middle_index]
        data_y_test = data_y[middle_index:]

        return data_x_train, data_y_train, data_x_test, data_y_test

    def generate_loader(self):

        data_x, data_y, train_range_list = Dataloader.preprocess(self)

        data_x_train, data_y_train, data_x_test, data_y_test = Dataloader.split_x_y_list(self,data_x, data_y)

        my_x_train = np.array(data_x_train)
        my_x_test = np.array(data_x_test)

        my_y_train = np.array(data_y_train)
        my_y_test = np.array(data_y_test)

        tensor_x_train = torch.Tensor(my_x_train).to(self.device)
        tensor_x_test = torch.Tensor(my_x_test).to(self.device)

        tensor_x_train = Dataloader.pool(self, tensor_x_train)
        tensor_x_test = Dataloader.pool(self, tensor_x_test) if tensor_x_test.size(dim = 0) > 0 else tensor_x_test

        tensor_y_train = torch.Tensor(my_y_train).to(self.device)
        tensor_y_test = torch.Tensor(my_y_test).to(self.device)

        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

        train_loader = DataLoader(train_dataset, batch_size = self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size)

        if self.print == True:
            print('Training Set:\n')
            for idx, (image_x, image_y) in enumerate(train_loader):
                print(idx, 'Image x batch dimensions:', image_x.size(), 'Image y batch dimensions:', image_y.size())
            print('\nTesting Set:')
            for idx, (image_x, image_y) in enumerate(test_loader):
                print(idx, 'Image x batch dimensions:', image_x.size(), 'Image y batch dimensions:', image_y.size())

        return train_loader, test_loader, train_range_list