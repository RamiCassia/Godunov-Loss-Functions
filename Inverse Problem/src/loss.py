import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

from src.loss_components import Conv1dDerivative, Conv2dDerivative, HLLC

import torch
import torch.nn as nn

class loss_generator(nn.Module):

    def __init__(self, dt, dx, dy, gamma, NX, NY, loss_type, device, e_tvd, e_ent, e_visc, scale_factor, pool_mode):

        super(loss_generator, self).__init__()

        self.scale_factor = scale_factor
        self.pool_mode = pool_mode
        self.gamma = gamma
        self.loss_type = loss_type

        self.partial_t = [[[-1, 1]]]

        self.partial_x = [[[[ 0, 0, 0],
                            [-1/2, 0, 1/2],
                            [ 0, 0, 0]]]]

        self.partial_y = [[[[0, -1/2, 0],
                            [0,  0, 0],
                            [0,  1/2, 0]]]]

        self.NX = NX
        self.NY = NY
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.cfl_x = self.dt/self.dx
        self.cfl_y = self.dt/self.dy
        self.device = device
        self.e_tvd = e_tvd
        self.e_ent = e_ent
        self.e_visc = e_visc

        self.ddt = Conv1dDerivative(DerFilter = self.partial_t, resol = (self.dt), kernel_size = 3, name = 'partial_t').to(self.device)
        self.ddx = Conv2dDerivative(DerFilter = self.partial_x, resol = (self.dx), kernel_size = 3, name = 'dx_operator').to(self.device)
        self.ddy = Conv2dDerivative(DerFilter = self.partial_y, resol = (self.dy), kernel_size = 3, name = 'dy_operator').to(self.device)

    def get_phy_Loss_GOD(self, output):

        output_m = output.clone()

        output_m[:,:,3,:,:] = output[:,:,3,:,:].clone() + 0.5*(output[:,:,1,:,:].clone()*output[:,:,1,:,:].clone() + output[:,:,2,:,:].clone()*output[:,:,2,:,:].clone())

        Q = output_m.clone()
        U = output_m.clone()

        delta_flux_y, delta_flux_x = HLLC.flux_hllc(output_m.clone(), self.NX, self.NY, self.gamma)

        delta_flux_y, delta_flux_x = delta_flux_y[:,:-1,:,:,:], delta_flux_x[:,:-1,:,:,:]


        Q[:,:,1,:,:] = Q[:,:,1,:,:].clone() * U[:,:,0,:,:]
        Q[:,:,2,:,:] = Q[:,:,2,:,:].clone() * U[:,:,0,:,:]
        Q[:,:,3,:,:] = Q[:,:,3,:,:].clone() * U[:,:,0,:,:]

        Q_t = (Q[:,1:,:,:,:] - Q[:,:-1,:,:,:])/self.dt

        flux_x = (1/self.dx)*(delta_flux_x[:,:,:,:,:])
        flux_y = (1/self.dy)*(delta_flux_y[:,:,:,:,:])

        f_Q = Q_t[:,:,:,:,:].to(self.device) + flux_x.to(self.device) + flux_y.to(self.device)

        return f_Q

    def get_phy_Loss_LF(self, output):

        U = output.clone()
        Q = output.clone()

        U[:,:,3,:,:] = U[:,:,3,:,:].clone() + 0.5*(U[:,:,1,:,:].clone()*U[:,:,1,:,:].clone() + U[:,:,2,:,:].clone()*U[:,:,2,:,:].clone())
        Q[:,:,3,:,:] = Q[:,:,3,:,:].clone() + 0.5*(Q[:,:,1,:,:].clone()*Q[:,:,1,:,:].clone() + Q[:,:,2,:,:].clone()*Q[:,:,2,:,:].clone())


        p = U[:,:,0:1,:,:] * (self.gamma - 1) * (U[:,:,3:4,:,:] - 0.5*((U[:,:,1:2,:,:]*U[:,:,1:2,:,:]) + (U[:,:,2:3,:,:]*U[:,:,2:3,:,:])))

        F1 = U[:,:,0:1,:,:]*U[:,:,1:2,:,:]
        F2 = U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,1:2,:,:] + p
        F3 = U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,2:3,:,:]
        F4 = (U[:,:,0:1,:,:]*U[:,:,3:4,:,:] + p)*U[:,:,1:2,:,:]
        F = torch.cat((F1, F2, F3, F4), dim=2)

        G1 = U[:,:,0:1,:,:]*U[:,:,2:3,:,:]
        G2 = U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,2:3,:,:]
        G3 = U[:,:,0:1,:,:]*U[:,:,2:3,:,:]*U[:,:,2:3,:,:] + p
        G4 = (U[:,:,0:1,:,:]*U[:,:,3:4,:,:] + p)*U[:,:,2:3,:,:]
        G = torch.cat((G1, G2, G3, G4), dim=2)

        Q1 = U[:,:,0:1,:,:]
        Q2 = U[:,:,1:2,:,:]*U[:,:,0:1,:,:]
        Q3 = U[:,:,2:3,:,:]*U[:,:,0:1,:,:]
        Q4 = U[:,:,3:4,:,:]*U[:,:,0:1,:,:]
        Q = torch.cat((Q1, Q2, Q3, Q4), dim=2)

        F_avg_m = 0.5*(F[:,:,:,1:-1,:-2] + F[:,:,:,1:-1,1:-1])
        G_avg_m = 0.5*(G[:,:,:,:-2,1:-1] + G[:,:,:,1:-1,1:-1])

        F_avg_p = 0.5*(F[:,:,:,1:-1,2:] + F[:,:,:,1:-1,1:-1])
        G_avg_p = 0.5*(G[:,:,:,2:,1:-1] + G[:,:,:,1:-1,1:-1])

        F_tilde_m = F_avg_m - 0.5*(self.dx/self.dt)*(Q[:,:,:,1:-1,1:-1] - Q[:,:,:,1:-1,:-2])
        G_tilde_m = G_avg_m - 0.5*(self.dy/self.dt)*(Q[:,:,:,1:-1,1:-1] - Q[:,:,:,:-2,1:-1])

        F_tilde_p = F_avg_p - 0.5*(self.dx/self.dt)*(Q[:,:,:,1:-1,2:] - Q[:,:,:,1:-1,1:-1])
        G_tilde_p = G_avg_p - 0.5*(self.dy/self.dt)*(Q[:,:,:,2:,1:-1] - Q[:,:,:,1:-1,1:-1])

        flux_x = (1/self.dx)*(F_tilde_p - F_tilde_m)
        flux_y = (1/self.dy)*(G_tilde_p - G_tilde_m)

        flux_x_final = flux_x[:,-1:,:,:,:]
        flux_y_final = flux_y[:,-1:,:,:,:]

        Q_t = (Q[:,1:,:,1:-1,1:-1] - Q[:,:-1,:,1:-1,1:-1])/self.dt

        f_Q = Q_t.to(self.device) + flux_x_final.to(self.device) + flux_y_final.to(self.device)

        return f_Q


    def get_phy_Loss_PINN(self, output):

        U = output.clone()
        Q = output.clone()

        U[:,:,3,:,:] = U[:,:,3,:,:].clone() + 0.5*(U[:,:,1,:,:].clone()*U[:,:,1,:,:].clone() + U[:,:,2,:,:].clone()*U[:,:,2,:,:].clone())
        Q[:,:,3,:,:] = Q[:,:,3,:,:].clone() + 0.5*(Q[:,:,1,:,:].clone()*Q[:,:,1,:,:].clone() + Q[:,:,2,:,:].clone()*Q[:,:,2,:,:].clone())

        p = U[:,:,0:1,:,:] * (self.gamma - 1) * (U[:,:,3:4,:,:] - 0.5*((U[:,:,1:2,:,:]*U[:,:,1:2,:,:]) + (U[:,:,2:3,:,:]*U[:,:,2:3,:,:])))

        F1_x = self.ddx(U[:,:,0:1,:,:]*U[:,:,1:2,:,:])
        F2_x = self.ddx(U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,1:2,:,:] + p)
        F3_x = self.ddx(U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,2:3,:,:])
        F4_x = self.ddx((U[:,:,0:1,:,:]*U[:,:,3:4,:,:] + p)*U[:,:,1:2,:,:])
        F_x = torch.cat((F1_x, F2_x, F3_x, F4_x), dim=2)

        G1_y = self.ddy(U[:,:,0:1,:,:]*U[:,:,2:3,:,:])
        G2_y = self.ddy(U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*U[:,:,2:3,:,:])
        G3_y = self.ddy(U[:,:,0:1,:,:]*U[:,:,2:3,:,:]*U[:,:,2:3,:,:] + p)
        G4_y = self.ddy((U[:,:,0:1,:,:]*U[:,:,3:4,:,:] + p)*U[:,:,2:3,:,:])
        G_y = torch.cat((G1_y, G2_y, G3_y, G4_y), dim=2)

        Q1_xx = self.ddx(self.ddx(U[:,:,0:1,:,:]))
        Q2_xx = self.ddx(self.ddx(U[:,:,1:2,:,:]*U[:,:,0:1,:,:]))
        Q3_xx = self.ddx(self.ddx(U[:,:,2:3,:,:]*U[:,:,0:1,:,:]))
        Q4_xx = self.ddx(self.ddx(U[:,:,3:4,:,:]*U[:,:,0:1,:,:]))
        Q_xx = torch.cat((Q1_xx, Q2_xx, Q3_xx, Q4_xx), dim=2)

        Q1_yy = self.ddy(self.ddy(U[:,:,0:1,:,:]))
        Q2_yy = self.ddy(self.ddy(U[:,:,1:2,:,:]*U[:,:,0:1,:,:]))
        Q3_yy = self.ddy(self.ddy(U[:,:,2:3,:,:]*U[:,:,0:1,:,:]))
        Q4_yy = self.ddy(self.ddy(U[:,:,3:4,:,:]*U[:,:,0:1,:,:]))
        Q_yy = torch.cat((Q1_yy, Q2_yy, Q3_yy, Q4_yy), dim=2)

        Q[:,:,1,:,:] = Q[:,:,1,:,:].clone() * U[:,:,0,:,:]
        Q[:,:,2,:,:] = Q[:,:,2,:,:].clone() * U[:,:,0,:,:]
        Q[:,:,3,:,:] = Q[:,:,3,:,:].clone() * U[:,:,0,:,:]

        Q_t = (Q[:,1:,:,:,:] - Q[:,:-1,:,:,:])/self.dt

        Q_t = Q_t[:,:,:,:,:]
        F_x = F_x[:,:-1,:,:,:]
        G_y = G_y[:,:-1,:,:,:]

        lap = Q_xx[:,:-1,:,:,:] + Q_yy[:,:-1,:,:,:]

        f_Q = Q_t + F_x + G_y - self.e_visc*lap

        return f_Q

    def forward(self, input, output, interp, target, reg_weight):

        mse_loss = nn.MSELoss()

        U = output.clone()

        U[:,:,3,:,:] = U[:,:,3,:,:].clone() + 0.5*(U[:,:,1,:,:].clone()*U[:,:,1,:,:].clone() + U[:,:,2,:,:].clone()*U[:,:,2,:,:].clone())

        p = U[:,:,0:1,:,:] * (self.gamma - 1) * (U[:,:,3:4,:,:] - 0.5*((U[:,:,1:2,:,:]*U[:,:,1:2,:,:]) + (U[:,:,2:3,:,:]*U[:,:,2:3,:,:])))
        U_tild = -U[:,:,0:1,:,:]*torch.log(p/(self.gamma * U[:,:,0:1,:,:]))
        F_tild_dx = self.ddx(-U[:,:,0:1,:,:]*U[:,:,1:2,:,:]*torch.log(p/(self.gamma * U[:,:,0:1,:,:])))
        G_tild_dy = self.ddy(-U[:,:,0:1,:,:]*U[:,:,2:3,:,:]*torch.log(p/(self.gamma * U[:,:,0:1,:,:])))
        ent = (U_tild[:,1:,:,:,:] - U_tild[:,:-1,:,:,:])/self.dt + F_tild_dx[:,:-1,:,:,:] + G_tild_dy[:,:-1,:,:,:]
        ent_max = torch.maximum(ent, torch.zeros_like(ent))
        ent_loss = mse_loss(ent_max, torch.zeros_like(ent_max))

        TV  = torch.sum(self.dx*torch.abs(U[:,:,:,1:,1:] - U[:,:,:,:-1,1:]) + self.dy*torch.abs(U[:,:,:,1:,1:] - U[:,:,:,1:,:-1]), dim = (2,3,4))
        TV_max = torch.maximum(TV[:,1:] - TV[:,:-1], torch.zeros_like(TV[:,1:]))
        TV_loss = mse_loss(TV_max, torch.zeros_like(TV_max))

        if self.loss_type == 'GOD':
            f_Q = loss_generator.get_phy_Loss_GOD(self, output)
        elif self.loss_type =='PINN':
            f_Q = loss_generator.get_phy_Loss_PINN(self, output)
        elif self.loss_type =='LAX':
            f_Q = loss_generator.get_phy_Loss_LF(self, output)

        output_pool = output.clone()
        a, b, c, d ,e = output.shape
        output_pool = output_pool.reshape(a*b,c,d,e)

        if self.pool_mode == 'max':

            output_pool = F.max_pool2d(output_pool, (self.scale_factor, self.scale_factor))
        else:
            output_pool = F.avg_pool2d(output_pool, (self.scale_factor, self.scale_factor))

        output_pool = output_pool.reshape(a,b,c,int(d/self.scale_factor),int(e/self.scale_factor))
        reg_loss = reg_weight*torch.norm(input - output_pool)

        loss = mse_loss(f_Q[:,:,:,:,:], torch.zeros_like(f_Q[:,:,:,:,:])) + reg_loss + self.e_tvd*TV_loss + self.e_ent*ent_loss

        return loss, reg_loss