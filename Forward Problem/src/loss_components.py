import torch
import torch.nn as nn

class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, padding=1, padding_mode = 'replicate', bias=False)
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        batch = input.size(dim = 0)
        time = input.size(dim = 1)
        height = input.size(dim = 3)
        width = input.size(dim = 4)

        input=input.reshape(batch*time, 1, height, width)
        derivative = self.filter(input)
        derivative = derivative.reshape(batch, time, 1, height, width)
        return derivative / self.resol

class HLLC():

    def get_U_From_Cons(rho, rho_u, rho_v, rho_E):
        return torch.stack((rho, rho_u, rho_v, rho_E))

    def get_Cons_From_U(X, nx, ny):
        Xresh = X.permute(0, 1, 3, 4, 2)
        rho = Xresh[:,:,:,:,0]
        rho_u = Xresh[:,:,:,:,1]
        rho_v = Xresh[:,:,:,:,2]
        rho_E = Xresh[:,:,:,:,3]
        return rho, rho_u, rho_v, rho_E

    def computeOtherVariables(rho, rho_u, rho_v, rho_E, gamma=1.4):
        u = rho_u.clone() / rho.clone()
        v = rho_v.clone() / rho.clone()
        E = rho_E.clone() / rho.clone()
        a = torch.sqrt(gamma * (gamma - 1) * (E - 0.5 * (u * u + v * v)))
        p = rho * (gamma - 1) * (E - 0.5 * (u * u + v * v))
        return {'r': rho, 'u': u, 'v': v, 'E': E, 'a': a, 'p': p}

    def fluxEulerPhysique(W, direction, gamma=1.4):
        rho = W[:,:,0,:]
        rho_u = W[:,:,1,:]
        rho_v = W[:,:,2,:]
        rho_E = W[:,:,3,:]

        out = HLLC.computeOtherVariables(rho, rho_u, rho_v, rho_E)
        u, v, p = out['u'], out['v'], out['p']

        F = torch.zeros_like(W)

        if direction == 0:
            F[:,:,0,:] = rho_u
            F[:,:,1,:] = rho_u.clone() * u + p
            F[:,:,2,:] = rho_u.clone() * v
            F[:,:,3,:] = (rho_E.clone() + p) * u
        elif direction == 1:
            F[:,:,0,:] = rho_v
            F[:,:,1,:] = rho_v * u
            F[:,:,2,:] = rho_v * v + p
            F[:,:,3,:] = (rho_E + p) * v

        return F

    def HLLCsolver(WL, WR, gamma=1.4):
        rhoL, rho_uL, rho_vL, rho_EL = WL[:,:,0,:], WL[:,:,1,:], WL[:,:,2,:], WL[:,:,3,:]
        rhoR, rho_uR, rho_vR, rho_ER = WR[:,:,0,:], WR[:,:,1,:], WR[:,:,2,:], WR[:,:,3,:]

        out = HLLC.computeOtherVariables(rhoR, rho_uR, rho_vR, rho_ER)
        uR, vR, pR, ER, aR = out['u'], out['v'], out['p'], out['E'], out['a']

        out = HLLC.computeOtherVariables(rhoL, rho_uL, rho_vL, rho_EL)
        uL, vL, pL, EL, aL = out['u'], out['v'], out['p'], out['E'], out['a']

        z_pow = (gamma - 1) / (2 * gamma)
        pstarbar = ((aL + aR - 0.5 * (gamma - 1) * (uR - uL)) / ((aL / (pL ** z_pow)) + (aR / (pR ** z_pow))))**(1/z_pow)

        qL = torch.where(pstarbar > pL, torch.sqrt(1 + ((gamma + 1) / (2 * gamma)) * ((pstarbar / pL) - 1)), 1.0)
        qR = torch.where(pstarbar > pR, torch.sqrt(1 + ((gamma + 1) / (2 * gamma)) * ((pstarbar / pR) - 1)), 1.0)
        SL = uL - aL * qL
        SR = uR + aR * qR

        Sstar = (pR - pL + rhoL.clone() * uL * (SL - uL) - rhoR.clone() * uR * (SR - uR)) / (rhoL.clone() * (SL - uL) - rhoR.clone() * (SR - uR))

        Wstar_L = torch.zeros_like(WL)
        coeff = rhoL.clone() * (SL - uL) / (SL - Sstar)
        Wstar_L[:,:,0,:] = coeff
        Wstar_L[:,:,1,:] = coeff * Sstar
        Wstar_L[:,:,2,:] = coeff * vL
        Wstar_L[:,:,3,:] = coeff * (EL + (Sstar - uL) * (Sstar + pL / (rhoL.clone() * (SL - uL))))

        Wstar_R = torch.zeros_like(WL)
        coeff = rhoR.clone() * (SR - uR) / (SR - Sstar)
        Wstar_R[:,:,0,:] = coeff
        Wstar_R[:,:,1,:] = coeff * Sstar
        Wstar_R[:,:,2,:] = coeff * vR
        Wstar_R[:,:,3,:] = coeff * (ER + (Sstar - uR) * (Sstar + pR / (rhoR.clone() * (SR - uR))))
        
        batch, time = SL.size(0), SL.size(1)
        stck_SL = SL.unsqueeze(2).expand(batch, time, 4, -1)
        stck_Sstar = Sstar.unsqueeze(2).expand(batch, time, 4, -1)
        stck_SR = SR.unsqueeze(2).expand(batch, time, 4, -1)

        face_flux1 = torch.where(stck_SL>0, HLLC.fluxEulerPhysique(WL,direction=0), 0)
        face_flux2 = torch.where((stck_SL<=0) & (stck_Sstar>=0), HLLC.fluxEulerPhysique(Wstar_L,direction=0), 0)
        face_flux3 = torch.where((stck_SR>0) & (stck_Sstar<0), HLLC.fluxEulerPhysique(Wstar_R,direction=0), 0)
        face_flux4 = torch.where(stck_SR<=0, HLLC.fluxEulerPhysique(WR,direction=0), 0)

        face_flux = face_flux1 + face_flux2 + face_flux3 + face_flux4


        return face_flux

    def computeFluxes(WL, WR, direction, gamma=1.4):
        if direction == 1:
            WR[:,:,[1,2],:] = WR[:,:,[2,1],:]
            WL[:,:,[1,2],:] = WL[:,:,[2,1],:]

        face_flux = HLLC.HLLCsolver(WL, WR)

        if direction == 1:
            face_flux[:,:,[1,2],:] = face_flux[:,:,[2,1],:]
            WR[:,:,[1,2],:] = WR[:,:,[2,1],:]
            WL[:,:,[1,2],:] = WL[:,:,[2,1],:]

        return face_flux


    def flux_hllc(U, nx, ny, gamma=1.4):
        
        batch, time = U.size(0), U.size(1)

        U[:,:,1,:,:] = U[:,:,0,:,:].clone() * U[:,:,1,:,:].clone()
        U[:,:,2,:,:] = U[:,:,0,:,:].clone() * U[:,:,2,:,:].clone()
        U[:,:,3,:,:] = U[:,:,0,:,:].clone() * U[:,:,3,:,:].clone()

        rho, rho_u, rho_v, rho_E = HLLC.get_Cons_From_U(U, nx, ny)

        Wup = torch.zeros((batch, time, 4, (nx) * (ny - 1)))
        Wup[:,:,0,:] = rho[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,1,:] = rho_u[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,2,:] = rho_v[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,3,:] = rho_E[:,:,:-1,:].reshape(batch, time, -1)

        Wdown = torch.zeros((batch, time, 4, (nx) * (ny - 1)))
        Wdown[:,:,0,:] = rho[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,1,:] = rho_u[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,2,:] = rho_v[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,3,:] = rho_E[:,:,1:,:].reshape(batch, time, -1)

        Wleft = torch.zeros((batch, time, 4, (nx - 1) * (ny)))
        Wleft[:,:,0,:] = rho[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,1,:] = rho_u[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,2,:] = rho_v[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,3,:] = rho_E[:,:,:,:-1].reshape(batch, time, -1)

        Wright = torch.zeros((batch, time, 4, (nx - 1) * (ny)))
        Wright[:,:,0,:] = rho[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,1,:] = rho_u[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,2,:] = rho_v[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,3,:] = rho_E[:,:,:,1:].reshape(batch, time, -1)

        fluxes_down = torch.zeros((batch, time, 4, ny + 1, nx))
        fluxes_down_inner = HLLC.computeFluxes(Wup, Wdown, direction=1)
        fluxes_down[:,:,:,1:-1,:] = fluxes_down_inner.reshape(batch, time, 4, ny - 1, nx)

        fluxes_right = torch.zeros((batch, time, 4, ny, nx + 1))
        fluxes_right_inner = HLLC.computeFluxes(Wleft, Wright, direction=0)
        fluxes_right[:,:,:,:,1:-1] = fluxes_right_inner.reshape(batch, time, 4, ny, nx - 1)

        W = torch.zeros((batch, time, 4, ny))
        W[:,:,0,:] = rho[:,:,:,0]
        W[:,:,1,:] = rho_u[:,:,:,0]
        W[:,:,2,:] = rho_v[:,:,:,0]
        W[:,:,3,:] = rho_E[:,:,:,0]

        fluxes_right_outer = HLLC.fluxEulerPhysique(W, direction=0)
        fluxes_right[:,:,:,:,0] = fluxes_right_outer.reshape(batch, time, 4, ny)

        W = torch.zeros((batch, time, 4, ny))
        W[:,:,0,:] = rho[:,:,:,-1]
        W[:,:,1,:] = rho_u[:,:,:,-1]
        W[:,:,2,:] = rho_v[:,:,:,-1]
        W[:,:,3,:] = rho_E[:,:,:,-1]

        fluxes_right_outer = HLLC.fluxEulerPhysique(W, direction=0)
        fluxes_right[:,:,:,:,-1] = fluxes_right_outer.reshape(batch, time, 4, ny)

        W = torch.zeros((batch, time, 4, nx))
        W[:,:,0,:] = rho[:,:,0,:]
        W[:,:,1,:] = rho_u[:,:,0,:]
        W[:,:,2,:] = rho_v[:,:,0,:]
        W[:,:,3,:] = rho_E[:,:,0,:]

        fluxes_down_outer = HLLC.fluxEulerPhysique(W, direction=1)
        fluxes_down[:,:,:,0,:] = fluxes_down_outer.reshape(batch, time, 4, nx)

        W = torch.zeros((batch, time, 4, nx))
        W[:,:,0,:] = rho[:,:,-1,:]
        W[:,:,1,:] = rho_u[:,:,-1,:]
        W[:,:,2,:] = rho_v[:,:,-1,:]
        W[:,:,3,:] = rho_E[:,:,-1,:]

        fluxes_down_outer = HLLC.fluxEulerPhysique(W, direction=1)
        fluxes_down[:,:,:,-1,:] = fluxes_down_outer.reshape(batch, time, 4, nx)

        flux_diff_y = fluxes_down[:,:,:,1:,:] - fluxes_down[:,:,:,:-1,:]
        flux_diff_x = fluxes_right[:,:,:,:,1:] - fluxes_right[:,:,:,:,:-1]

        return flux_diff_y, flux_diff_x
    
