import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

from src.model_components import ConvLSTMCell, encoder_block, decoder_block, SC_output

import torch
import torch.nn as nn


class PhyCR_UNet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''
    def __init__(self, initial_states, depth, layer_type, input_channels, hidden_channels,
        input_kernel_size, input_stride, input_padding, input_dilation, padding_mode, activation, attention, batch_norm, dt,
        num_layers, step=1, effective_step=[1]):

        super(PhyCR_UNet, self).__init__()

        self.depth = depth
        self.layer_type = layer_type
        self.input_channels = [[input_channels] + hidden_channels[i][:] for i in range(self.depth)]
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.input_dilation = input_dilation
        self.padding_mode = padding_mode
        self.activation = activation
        self.attention = attention
        self.batch_norm = batch_norm
        self.dt = dt
        self.step = step
        self.effective_step = effective_step
        self.initial_states = initial_states

        self.num_encoder = [num_layers[i][0] for i in range(self.depth)]
        self.num_convlstm = [num_layers[i][1] for i in range(self.depth)]
        self.num_decoder = [num_layers[i][2] for i in range(self.depth)]

        self._all_layers = []

        for i in range(self.depth):
            for j in range(self.num_encoder[i]):
                name = 'encoder{}{}'.format(i,j)
                cell = encoder_block(
                    layer_type = self.layer_type[i][j],
                    input_channels = self.input_channels[i][j],
                    hidden_channels = self.hidden_channels[i][j],
                    input_kernel_size = self.input_kernel_size[i][j],
                    input_stride = self.input_stride[i][j],
                    input_padding = self.input_padding[i][j],
                    input_dilation = self.input_dilation[i][j],
                    padding_mode = self.padding_mode,
                    activation = self.activation,
                    batch_norm = self.batch_norm[i][j])

                setattr(self, name, cell)
                self._all_layers.append(cell)

        for i in range(self.depth):
            for j in range(self.num_encoder[i], self.num_encoder[i] + self.num_convlstm[i]):
                name = 'convlstm{}{}'.format(i,j)
                cell = ConvLSTMCell(
                    layer_type = self.layer_type[i][j],
                    input_channels = self.input_channels[i][j],
                    hidden_channels = self.hidden_channels[i][j],
                    input_kernel_size = self.input_kernel_size[i][j],
                    input_stride = self.input_stride[i][j],
                    input_padding = self.input_padding[i][j],
                    input_dilation = self.input_dilation[i][j],
                    padding_mode = self.padding_mode,
                    activation = self.activation,
                    attention = self.attention,
                    batch_norm = self.batch_norm[i][j])

                setattr(self, name, cell)
                self._all_layers.append(cell)

        for i in range(self.depth):
            for j in range(self.num_encoder[i] + self.num_convlstm[i], self.num_encoder[i] + self.num_convlstm[i] + self.num_decoder[i]):
                name = 'decoder{}{}'.format(i,j)
                cell = decoder_block(
                    layer_type = self.layer_type[i][j],
                    input_channels = self.input_channels[i][j],
                    hidden_channels = self.hidden_channels[i][j],
                    input_kernel_size = self.input_kernel_size[i][j],
                    input_stride = self.input_stride[i][j],
                    input_padding = self.input_padding[i][j],
                    input_dilation = self.input_dilation[i][j],
                    padding_mode = self.padding_mode,
                    activation = self.activation,
                    batch_norm = self.batch_norm[i][j])

                setattr(self, name, cell)
                self._all_layers.append(cell)

        for i in range(self.depth):
            for j in range(self.num_encoder[i] + self.num_convlstm[i], self.num_encoder[i] + self.num_convlstm[i] + 1):
                name = 'sc_output{}{}'.format(i,j)
                cell = SC_output(
                    layer_type = self.layer_type[i][j],
                    input_channels = self.input_channels[i][j],
                    hidden_channels = self.hidden_channels[i][j],
                    input_kernel_size = self.input_kernel_size[i][j],
                    input_stride = self.input_stride[i][j],
                    input_padding = self.input_padding[i][j],
                    input_dilation = self.input_dilation[i][j],
                    padding_mode = self.padding_mode,
                    activation = self.activation,
                    batch_norm = self.batch_norm[i][j])

                setattr(self, name, cell)
                self._all_layers.append(cell)

    @torch.jit._script_if_tracing 
    def forward(self, x):

        batch = x.size(dim = 0)
        time = x.size(dim = 1)
        channel = x.size(dim = 2)
        height = x.size(dim = 3)
        width = x.size(dim = 4)

        internal_states = [[] for i in range(self.depth)]
        outputs = []
        second_last_states = [[] for i in range(self.depth)]

        for step in range(self.step):

            stored_x_list = []
            xt = x
            x=x.reshape(batch*time, channel, height, width)
            x_LSTM = []
            x_decoder = []

            for i in range(self.depth - 1, -1, -1):

                if i == self.depth - 1:

                    for j in range(self.num_encoder[i]):
                        name = 'encoder{}{}'.format(i,j)

                        if getattr(self, name).layer_type != 'CR':
                            stored_x = x.clone()
                            stored_x_list.append(stored_x)

                        x = getattr(self, name)(x)

                else:
                    x = stored_x_list.pop()

                for j in range(self.num_encoder[i], self.num_encoder[i] + self.num_convlstm[i]):
                    name = 'convlstm{}{}'.format(i,j)

                    if step == 0:
                        (h, c, m) = getattr(self, name).init_hidden_tensor(prev_state = self.initial_states[i][j - self.num_encoder[i]])
                        internal_states[i].append((h, c, m))

                    (h, c, m) = internal_states[i][j - self.num_encoder[i]]
            
                    a = torch.cat((x,h,c,m), dim = 0)                    
                  
                    h, c, m, x = getattr(self, name)(a)
                    
                    internal_states[i][j - self.num_encoder[i]] = (h, c, m)

                x_LSTM = x.clone()

                if i < self.depth - 1:
                    x = torch.cat((x_LSTM, x_decoder), 1)


                    for j in range(self.num_encoder[i] + self.num_convlstm[i], self.num_encoder[i] + self.num_convlstm[i] + 1):
                        name = 'sc_output{}{}'.format(i,j)
                        x = getattr(self, name)(x)

                for j in range(self.num_encoder[i] + self.num_convlstm[i], self.num_encoder[i] + self.num_convlstm[i] + self.num_decoder[i]):

                    name = 'decoder{}{}'.format(i,j)
                    x = getattr(self, name)(x)

                x_decoder = x.clone()

         
            x = x.reshape(batch, time, channel, height, width)

            x = xt + self.dt * x

            x[:, :, [0, 3], :, :] = torch.abs((x[:, :, [0, 3], :, :].clone()))

            if step == (self.step - 2):
                second_last_states = internal_states.copy()

            if step in self.effective_step:
                outputs.append(x)

        return torch.cat(tuple(outputs), dim=1), second_last_states