import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
        input_stride, input_padding, input_dilation, padding_mode, layer_type, activation, attention, batch_norm):

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.padding_mode = padding_mode
        self.input_dilation = input_dilation
        self.layer_type = layer_type
        self.act = activation
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.attention = attention

        self.Wxi = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding = self.input_padding, dilation = self.input_dilation,
            bias=True, padding_mode=self.padding_mode)

        self.Whi = nn.Conv2d(in_channels = self.hidden_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding=self.input_padding, dilation = self.input_dilation, bias=False,
            padding_mode=self.padding_mode)

        self.Wxf = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding = self.input_padding, dilation = self.input_dilation,
            bias=True, padding_mode=self.padding_mode)

        self.Whf = nn.Conv2d(in_channels = self.hidden_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding=self.input_padding, dilation = self.input_dilation, bias=False,
            padding_mode=self.padding_mode)

        self.Wxc = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding =self.input_padding, dilation = self.input_dilation,
            bias=True, padding_mode=self.padding_mode)

        self.Whc = nn.Conv2d(in_channels = self.hidden_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding=self.input_padding, dilation = self.input_dilation, bias=False,
            padding_mode=self.padding_mode)

        self.Wxo = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding = self.input_padding, dilation = self.input_dilation,
            bias=True, padding_mode=self.padding_mode)

        self.Who = nn.Conv2d(in_channels = self.hidden_channels, out_channels = self.hidden_channels,
            kernel_size = self.input_kernel_size, stride = self.input_stride, padding=self.input_padding, dilation = self.input_dilation, bias=False,
            padding_mode=self.padding_mode)

        self.conv = weight_norm(nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels, kernel_size = self.input_kernel_size, stride = self.input_stride, dilation = self.input_dilation, padding = self.input_padding, padding_mode = 'replicate', bias=True))

        if attention == 'SA':
            self.query_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels, kernel_size = 1)
            self.key_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels, kernel_size = 1)
            self.value_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels, kernel_size = 1)
            self.z = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels, kernel_size = 1)

        if attention == 'SAM':
            self.query_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels, kernel_size =  1)
            self.key_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,  kernel_size =  1)
            self.value_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels,  kernel_size =  1)
            self.z_h = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels,  kernel_size =  1)

            self.key_m = nn.Conv2d(in_channels = self.input_channels, out_channels = self.hidden_channels,  kernel_size =  1)
            self.value_m = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels,  kernel_size =  1)
            self.z_m = nn.Conv2d(in_channels = self.input_channels, out_channels = self.input_channels, kernel_size =  1)
            self.w_z = nn.Conv2d(in_channels = self.input_channels*2, out_channels = self.input_channels*2, kernel_size =  1)
            self.w = nn.Conv2d(in_channels = self.input_channels*3, out_channels = self.input_channels*3,  kernel_size =  1)


    def forward(self, x):
        x1 = x.clone()
        sze = int(x1.size(0)/4)
        
        h = x1[sze:sze*2].clone()
        c = x1[sze*2:sze*3].clone()
        m = x1[sze*3:sze*4].clone()
        x =  x1[0:sze].clone()
    
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        new_c = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(new_c)


        if self.attention == 'SA':
            batch_size, _, H, W = ch.shape
            k_h = self.key_h(ch)
            q_h = self.query_h(ch)
            v_h = self.value_h(ch)

            k_h = k_h.view(batch_size, self.hidden_channels, H * W)
            q_h = q_h.view(batch_size, self.hidden_channels, H * W).transpose(1, 2)
            v_h = v_h.view(batch_size, self.input_channels, H * W)

            attention = torch.softmax(torch.bmm(q_h, k_h), dim=-1)

            new_H = torch.matmul(attention, v_h.permute(0, 2, 1))
            new_H = new_H.transpose(1, 2).view(batch_size, self.input_channels, H, W)
            new_H = self.z(new_H)
            new_M = m

        elif self.attention == 'SAM':
            batch_size, _, H, W = ch.shape

            k_h = self.key_h(ch)
            q_h = self.query_h(ch)
            v_h = self.value_h(ch)
            k_h = k_h.view(batch_size, self.hidden_channels, H * W)
            q_h = q_h.view(batch_size, self.hidden_channels, H * W).transpose(1, 2)
            v_h = v_h.view(batch_size, self.input_channels, H * W)

            attention_h = torch.softmax(torch.bmm(q_h, k_h), dim=-1)  # The shape is (batch_size, H*W, H*W)
            z_h = torch.matmul(attention_h, v_h.permute(0, 2, 1))
            z_h = z_h.transpose(1, 2).view(batch_size, self.input_channels, H, W)
            z_h = self.z_h(z_h)

            k_m = self.key_m(m)
            v_m = self.value_m(m)

            k_m = k_m.view(batch_size, self.hidden_channels, H * W) #1
            v_m = v_m.view(batch_size, self.input_channels, H * W) #1

            attention_m = torch.softmax(torch.bmm(q_h, k_m), dim=-1)
            z_m = torch.matmul(attention_m, v_m.permute(0, 2, 1))
            z_m = z_m.transpose(1, 2).view(batch_size, self.input_channels, H, W)
            z_m = self.z_m(z_m)

            Z = torch.cat([z_h, z_m], dim=1)
            Z = self.w_z(Z)
            W = torch.cat([Z, ch], dim=1)
            W = self.w(W)

            mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=3, dim=1)
            input_gate = torch.sigmoid(mi_conv)
            g = torch.tanh(mg_conv)
            new_M = (1 - input_gate) * m + input_gate * g
            output_gate = torch.sigmoid(mo_conv)
            new_H = output_gate * new_M
        else:
            new_H = ch
            new_M = m


        if self.batch_norm == True:
            return new_H, new_c, new_M, self.bn(self.conv(new_H)) if self.layer_type == 'LSTM_F' else self.act(self.bn(self.conv(new_H)))
        else:
            return new_H, new_c, new_M, self.conv(new_H) if self.layer_type == 'LSTM_F' else self.act(self.conv(new_H))

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda(), Variable(prev_state[2]).cuda())


class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
        input_stride, input_padding, input_dilation, padding_mode, layer_type, activation, batch_norm):

        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.input_dilation = input_dilation
        self.padding_mode = padding_mode
        self.layer_type = layer_type
        self.act = activation
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.max_pool = nn.MaxPool2d(kernel_size = self.input_kernel_size, stride = self.input_stride)
        self.avr_pool = nn.AvgPool2d(kernel_size = self.input_kernel_size, stride = self.input_stride)
        self.conv = weight_norm(nn.Conv2d(in_channels = self.input_channels,
            out_channels = self.hidden_channels, kernel_size = self.input_kernel_size, stride = self.input_stride, dilation = self.input_dilation, padding = self.input_padding, padding_mode= self.padding_mode, bias=True))

    def forward(self, x):

        if self.layer_type == 'CR' or self.layer_type == 'CRD':
            return self.act(self.bn(self.conv(x))) if self.batch_norm == True else self.act(self.conv(x))

        elif self.layer_type == 'MP':
            return self.max_pool(x)

        elif self.layer_type == 'AP':
            return self.avr_pool(x)


class decoder_block(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
        input_stride, input_padding, input_dilation, padding_mode, layer_type, activation, batch_norm):

        super(decoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = (input_padding,)*4
        self.input_dilation = input_dilation
        self.padding_mode = padding_mode
        self.layer_type = layer_type
        self.act = activation
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.upsample = nn.Upsample(scale_factor = 2, mode='nearest')
        self.convT = weight_norm(nn.ConvTranspose2d(in_channels = self.input_channels,
            out_channels = self.hidden_channels, kernel_size = self.input_kernel_size, stride = self.input_stride, dilation = self.input_dilation,
             bias=True))


    def forward(self, x):

        if self.layer_type == 'CTR':
            x = F.pad(x, pad = self.input_padding, mode = self.padding_mode)
            x = self.convT(self.act(x))
            if self.batch_norm == True:
                x = self.bn(x)

            return x[:,:,2:-2,2:-2]

        elif self.layer_type == 'CTRU':
            x = F.pad(x, pad = self.input_padding, mode = self.padding_mode)
            x = self.convT(self.act(x))
            if self.batch_norm == True:
                x = self.bn(x)

            return x[:,:,3:-3,3:-3]

        elif self.layer_type == 'UP':
            return self.upsample(x)

class SC_output(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
        input_stride, input_padding, input_dilation, padding_mode, layer_type, activation, batch_norm):

        super(SC_output, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = (input_padding,)*4
        self.input_dilation = input_dilation
        self.padding_mode = padding_mode
        self.layer_type = layer_type
        self.act = activation
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.conv = weight_norm(nn.Conv2d(in_channels = self.input_channels*2,
            out_channels = self.hidden_channels, kernel_size = 3, stride = 1, dilation = 1, padding = 1, padding_mode = 'replicate', bias=True))

    def forward(self, x):
            return self.bn(self.conv(x)) if self.batch_norm == True else self.conv(x)