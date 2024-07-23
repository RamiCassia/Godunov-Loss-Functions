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
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
            1, padding=0, bias=False)

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

    def get_Cons(X, nx, ny):
        Xresh = X.permute(0, 1, 3, 4, 2)
        rho = Xresh[:,:,:,:,0]
        rho_u = Xresh[:,:,:,:,1]
        rho_v = Xresh[:,:,:,:,2]
        rho_E = Xresh[:,:,:,:,3]
        return rho, rho_u, rho_v, rho_E

    def get_uvpaHE(X, nx, ny):
        Xresh = X.permute(0, 1, 3, 4, 2)
        u = Xresh[:,:,:,:,0]
        v = Xresh[:,:,:,:,1]
        p = Xresh[:,:,:,:,2]
        a = Xresh[:,:,:,:,3]
        H = Xresh[:,:,:,:,4]
        E = Xresh[:,:,:,:,5]
        return u, v, p, a, H, E

    def computeOtherVariables(rho, rho_u, rho_v, rho_E, gamma=1.4):
        eps = 0
        u = rho_u.clone() / (rho.clone() + eps)
        v = rho_v.clone() / (rho.clone() + eps)
        E = rho_E.clone() / (rho.clone() + eps)
        p = rho * (gamma - 1) * (E - 0.5 * (u*u + v*v))

        return {'u': u, 'v': v, 'p': p}

    def fluxEulerPhysique(W, A, direction, gamma=1.4):
        rho = W[:,:,0,:]
        rho_u = W[:,:,1,:]
        rho_v = W[:,:,2,:]
        rho_E = W[:,:,3,:]

        u = A[:,:,0,:].clone()
        v = A[:,:,1,:].clone()
        p = torch.maximum(A[:,:,2,:].clone(), torch.zeros_like(A[:,:,2,:].clone()))

        F = torch.zeros_like(W)

        if direction == 0:
            F[:,:,0,:] = rho_u
            F[:,:,1,:] = rho_u.clone() * u.clone() + p.clone()
            F[:,:,2,:] = rho_u.clone() * v.clone()
            F[:,:,3,:] = (rho_E.clone() + p.clone()) * u.clone()
        elif direction == 1:
            F[:,:,0,:] = rho_v
            F[:,:,1,:] = rho_v * u.clone()
            F[:,:,2,:] = rho_v * v.clone() + p.clone()
            F[:,:,3,:] = (rho_E + p.clone()) * v.clone()

        return F

    def fluxEulerPhysique_star(W, direction, gamma=1.4):
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

    def HLLCsolver(WL, WR, AL, AR, gamma=1.4):
        eps = 0

        rhoL, rho_uL, rho_vL, rho_EL = WL[:,:,0,:], WL[:,:,1,:], WL[:,:,2,:], WL[:,:,3,:]
        rhoR, rho_uR, rho_vR, rho_ER = WR[:,:,0,:], WR[:,:,1,:], WR[:,:,2,:], WR[:,:,3,:]

        uL, vL, pL, aL, HL, EL = AL[:,:,0,:].clone(), AL[:,:,1,:].clone(), AL[:,:,2,:].clone(), AL[:,:,3,:].clone(), AL[:,:,4,:].clone(), AL[:,:,5,:].clone()
        uR, vR, pR, aR, HR, ER = AR[:,:,0,:].clone(), AR[:,:,1,:].clone(), AR[:,:,2,:].clone(), AR[:,:,3,:].clone(), AR[:,:,4,:].clone(), AR[:,:,5,:].clone()

        z_pow = (gamma - 1) / (2 * gamma)
        pstarbar = ((aL + aR - 0.5 * (gamma - 1) * (uR - uL)) / ((aL / (pL ** z_pow)) + (aR / (pR ** z_pow)) + eps))**(1/z_pow)

        qL = torch.where(pstarbar > pL, torch.sqrt(1 + ((gamma + 1) / (2 * gamma)) * ((pstarbar / pL) - 1)), 1.0)
        qR = torch.where(pstarbar > pR, torch.sqrt(1 + ((gamma + 1) / (2 * gamma)) * ((pstarbar / pR) - 1)), 1.0)

        SL = uL - aL * qL
        SR = uR + aR * qR

        Sstar = (pR - pL + rhoL.clone() * uL * (SL - uL) - rhoR.clone() * uR * (SR - uR)) / (rhoL.clone() * (SL - uL) - rhoR.clone() * (SR - uR) + eps)

        Wstar_L = torch.zeros_like(WL)
        coeff = rhoL.clone() * (SL - uL) / ((SL - Sstar) + eps)
        Wstar_L[:,:,0,:] = coeff
        Wstar_L[:,:,1,:] = coeff * Sstar
        Wstar_L[:,:,2,:] = coeff * vL
        Wstar_L[:,:,3,:] = coeff * (EL + (Sstar - uL) * (Sstar + pL / (rhoL.clone()*(SL - uL) + eps)))

        Wstar_R = torch.zeros_like(WL)
        coeff = rhoR.clone() * (SR - uR) / ((SR - Sstar) + eps)
        Wstar_R[:,:,0,:] = coeff
        Wstar_R[:,:,1,:] = coeff * Sstar
        Wstar_R[:,:,2,:] = coeff * vR
        Wstar_R[:,:,3,:] = coeff * (ER + (Sstar - uR) * (Sstar + pR / (rhoR.clone()*(SR - uR) + eps)))

        batch, time = SL.size(0), SL.size(1)
        stck_SL = SL.unsqueeze(2).expand(batch, time, 4, -1)
        stck_Sstar = Sstar.unsqueeze(2).expand(batch, time, 4, -1)
        stck_SR = SR.unsqueeze(2).expand(batch, time, 4, -1)

        face_flux1 = torch.where(stck_SL>0, HLLC.fluxEulerPhysique(WL, AL, direction=0), 0)
        face_flux2 = torch.where((stck_SL<=0) & (stck_Sstar>=0), HLLC.fluxEulerPhysique_star(Wstar_L,direction=0), 0)
        face_flux3 = torch.where((stck_SR>0) & (stck_Sstar<0), HLLC.fluxEulerPhysique_star(Wstar_R,direction=0), 0)
        face_flux4 = torch.where(stck_SR<=0, HLLC.fluxEulerPhysique(WR, AR, direction=0), 0)

        face_flux = face_flux1 + face_flux2 + face_flux3 + face_flux4


        return face_flux

    def computeFluxes(WL, WR, AL, AR, direction, gamma=1.4):
        if direction == 1:
            WR[:,:,[1,2],:] = WR[:,:,[2,1],:]
            WL[:,:,[1,2],:] = WL[:,:,[2,1],:]

            AR[:,:,[0,1],:] = AR[:,:,[1,0],:]
            AL[:,:,[0,1],:] = AL[:,:,[1,0],:]

        face_flux = HLLC.HLLCsolver(WL, WR, AL, AR)

        if direction == 1:
            face_flux[:,:,[1,2],:] = face_flux[:,:,[2,1],:]

            WR[:,:,[1,2],:] = WR[:,:,[2,1],:]
            WL[:,:,[1,2],:] = WL[:,:,[2,1],:]

            AR[:,:,[0,1],:] = AR[:,:,[1,0],:]
            AL[:,:,[0,1],:] = AL[:,:,[1,0],:]

        return face_flux


    def flux_hllc(Pri, nx, ny, gamma=1.4):

        batch, time = Pri.size(0), Pri.size(1)

        uvpaHE = torch.zeros(Pri.size(0), Pri.size(1), 6, nx, ny).cuda()
        uvpaHE[:,:,0,:,:] = Pri[:,:,1,:,:].clone()
        uvpaHE[:,:,1,:,:] = Pri[:,:,2,:,:].clone()
        uvpaHE[:,:,2,:,:] = Pri[:,:,0,:,:].clone()*(gamma - 1)*(Pri[:,:,3,:,:].clone() - 0.5*(Pri[:,:,1,:,:].clone()*Pri[:,:,1,:,:].clone() + Pri[:,:,2,:,:].clone()*Pri[:,:,2,:,:].clone()))
        uvpaHE[:,:,3,:,:] = torch.sqrt(gamma*(gamma - 1)*(Pri[:,:,3,:,:].clone() - 0.5*(Pri[:,:,1,:,:].clone()*Pri[:,:,1,:,:].clone() + Pri[:,:,2,:,:].clone()*Pri[:,:,2,:,:].clone())))
        uvpaHE[:,:,4,:,:] = 0.5*(Pri[:,:,1,:,:].clone()*Pri[:,:,1,:,:].clone() + Pri[:,:,2,:,:].clone()*Pri[:,:,2,:,:].clone()) + (uvpaHE[:,:,3,:,:].clone()*uvpaHE[:,:,3,:,:].clone())/(gamma - 1)
        uvpaHE[:,:,5,:,:] = Pri[:,:,3,:,:].clone()

        Con = Pri.clone()
        Con = Con.cuda()
        Con[:,:,1,:,:] = Pri[:,:,0,:,:].clone() * Pri[:,:,1,:,:].clone()
        Con[:,:,2,:,:] = Pri[:,:,0,:,:].clone() * Pri[:,:,2,:,:].clone()
        Con[:,:,3,:,:] = Pri[:,:,0,:,:].clone() * Pri[:,:,3,:,:].clone()

        rho, rho_u, rho_v, rho_E = HLLC.get_Cons(Con, nx, ny)
        u, v, p, a, H, E = HLLC.get_uvpaHE(uvpaHE, nx, ny)

        Wup = torch.zeros((batch, time, 4, (nx) * (ny - 1)))
        Wup[:,:,0,:] = rho[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,1,:] = rho_u[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,2,:] = rho_v[:,:,:-1,:].reshape(batch, time, -1)
        Wup[:,:,3,:] = rho_E[:,:,:-1,:].reshape(batch, time, -1)

        Aup = torch.zeros((batch, time, 6, (nx) * (ny - 1)))
        Aup[:,:,0,:] = u[:,:,:-1,:].reshape(batch, time, -1)
        Aup[:,:,1,:] = v[:,:,:-1,:].reshape(batch, time, -1)
        Aup[:,:,2,:] = p[:,:,:-1,:].reshape(batch, time, -1)
        Aup[:,:,3,:] = a[:,:,:-1,:].reshape(batch, time, -1)
        Aup[:,:,4,:] = H[:,:,:-1,:].reshape(batch, time, -1)
        Aup[:,:,5,:] = E[:,:,:-1,:].reshape(batch, time, -1)

        Wdown = torch.zeros((batch, time, 4, (nx) * (ny - 1)))
        Wdown[:,:,0,:] = rho[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,1,:] = rho_u[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,2,:] = rho_v[:,:,1:,:].reshape(batch, time, -1)
        Wdown[:,:,3,:] = rho_E[:,:,1:,:].reshape(batch, time, -1)

        Adown = torch.zeros((batch, time, 6, (nx) * (ny - 1)))
        Adown[:,:,0,:] = u[:,:,1:,:].reshape(batch, time, -1)
        Adown[:,:,1,:] = v[:,:,1:,:].reshape(batch, time, -1)
        Adown[:,:,2,:] = p[:,:,1:,:].reshape(batch, time, -1)
        Adown[:,:,3,:] = a[:,:,1:,:].reshape(batch, time, -1)
        Adown[:,:,4,:] = H[:,:,1:,:].reshape(batch, time, -1)
        Adown[:,:,5,:] = E[:,:,1:,:].reshape(batch, time, -1)

        Wleft = torch.zeros((batch, time, 4, (nx - 1) * (ny)))
        Wleft[:,:,0,:] = rho[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,1,:] = rho_u[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,2,:] = rho_v[:,:,:,:-1].reshape(batch, time, -1)
        Wleft[:,:,3,:] = rho_E[:,:,:,:-1].reshape(batch, time, -1)

        Aleft = torch.zeros((batch, time, 6, (nx - 1) * (ny)))
        Aleft[:,:,0,:] = u[:,:,:,:-1].reshape(batch, time, -1)
        Aleft[:,:,1,:] = v[:,:,:,:-1].reshape(batch, time, -1)
        Aleft[:,:,2,:] = p[:,:,:,:-1].reshape(batch, time, -1)
        Aleft[:,:,3,:] = a[:,:,:,:-1].reshape(batch, time, -1)
        Aleft[:,:,4,:] = H[:,:,:,:-1].reshape(batch, time, -1)
        Aleft[:,:,5,:] = E[:,:,:,:-1].reshape(batch, time, -1)

        Wright = torch.zeros((batch, time, 4, (nx - 1) * (ny)))
        Wright[:,:,0,:] = rho[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,1,:] = rho_u[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,2,:] = rho_v[:,:,:,1:].reshape(batch, time, -1)
        Wright[:,:,3,:] = rho_E[:,:,:,1:].reshape(batch, time, -1)

        Aright = torch.zeros((batch, time, 6, (nx - 1) * (ny)))
        Aright[:,:,0,:] = u[:,:,:,1:].reshape(batch, time, -1)
        Aright[:,:,1,:] = v[:,:,:,1:].reshape(batch, time, -1)
        Aright[:,:,2,:] = p[:,:,:,1:].reshape(batch, time, -1)
        Aright[:,:,3,:] = a[:,:,:,1:].reshape(batch, time, -1)
        Aright[:,:,4,:] = H[:,:,:,1:].reshape(batch, time, -1)
        Aright[:,:,5,:] = E[:,:,:,1:].reshape(batch, time, -1)

        fluxes_down = torch.zeros((batch, time, 4, ny + 1, nx))
        fluxes_down_inner = HLLC.computeFluxes(Wup, Wdown, Aup, Adown, direction=1)
        fluxes_down[:,:,:,1:-1,:] = fluxes_down_inner.reshape(batch, time, 4, ny - 1, nx)

        fluxes_right = torch.zeros((batch, time, 4, ny, nx + 1))
        fluxes_right_inner = HLLC.computeFluxes(Wleft, Wright, Aleft, Aright, direction=0)
        fluxes_right[:,:,:,:,1:-1] = fluxes_right_inner.reshape(batch, time, 4, ny, nx - 1)

        W = torch.zeros((batch, time, 4, ny))
        W[:,:,0,:] = rho[:,:,:,0]
        W[:,:,1,:] = rho_u[:,:,:,0]
        W[:,:,2,:] = rho_v[:,:,:,0]
        W[:,:,3,:] = rho_E[:,:,:,0]

        A = torch.zeros((batch, time, 6, ny))
        A[:,:,0,:] = u[:,:,:,0]
        A[:,:,1,:] = v[:,:,:,0]
        A[:,:,2,:] = p[:,:,:,0]
        A[:,:,3,:] = a[:,:,:,0]
        A[:,:,4,:] = H[:,:,:,0]
        A[:,:,5,:] = E[:,:,:,0]

        fluxes_right_outer = HLLC.fluxEulerPhysique(W, A, direction=0)
        fluxes_right[:,:,:,:,0] = fluxes_right_outer.reshape(batch, time, 4, ny)

        W = torch.zeros((batch, time, 4, ny))
        W[:,:,0,:] = rho[:,:,:,-1]
        W[:,:,1,:] = rho_u[:,:,:,-1]
        W[:,:,2,:] = rho_v[:,:,:,-1]
        W[:,:,3,:] = rho_E[:,:,:,-1]

        A = torch.zeros((batch, time, 6, ny))
        A[:,:,0,:] = u[:,:,:,-1]
        A[:,:,1,:] = v[:,:,:,-1]
        A[:,:,2,:] = p[:,:,:,-1]
        A[:,:,3,:] = a[:,:,:,-1]
        A[:,:,4,:] = H[:,:,:,-1]
        A[:,:,5,:] = E[:,:,:,-1]

        fluxes_right_outer = HLLC.fluxEulerPhysique(W, A, direction=0)
        fluxes_right[:,:,:,:,-1] = fluxes_right_outer.reshape(batch, time, 4, ny)

        W = torch.zeros((batch, time, 4, nx))
        W[:,:,0,:] = rho[:,:,0,:]
        W[:,:,1,:] = rho_u[:,:,0,:]
        W[:,:,2,:] = rho_v[:,:,0,:]
        W[:,:,3,:] = rho_E[:,:,0,:]

        A = torch.zeros((batch, time, 6, ny))
        A[:,:,0,:] = u[:,:,0,:]
        A[:,:,1,:] = v[:,:,0,:]
        A[:,:,2,:] = p[:,:,0,:]
        A[:,:,3,:] = a[:,:,0,:]
        A[:,:,4,:] = H[:,:,0,:]
        A[:,:,5,:] = E[:,:,0,:]

        fluxes_down_outer = HLLC.fluxEulerPhysique(W, A, direction=1)
        fluxes_down[:,:,:,0,:] = fluxes_down_outer.reshape(batch, time, 4, nx)

        W = torch.zeros((batch, time, 4, nx))
        W[:,:,0,:] = rho[:,:,-1,:]
        W[:,:,1,:] = rho_u[:,:,-1,:]
        W[:,:,2,:] = rho_v[:,:,-1,:]
        W[:,:,3,:] = rho_E[:,:,-1,:]

        A = torch.zeros((batch, time, 6, ny))
        A[:,:,0,:] = u[:,:,-1,:]
        A[:,:,1,:] = v[:,:,-1,:]
        A[:,:,2,:] = p[:,:,-1,:]
        A[:,:,3,:] = a[:,:,-1,:]
        A[:,:,4,:] = H[:,:,-1,:]
        A[:,:,5,:] = E[:,:,-1,:]

        fluxes_down_outer = HLLC.fluxEulerPhysique(W, A, direction=1)
        fluxes_down[:,:,:,-1,:] = fluxes_down_outer.reshape(batch, time, 4, nx)

        flux_diff_y = fluxes_down[:,:,:,1:,:] - fluxes_down[:,:,:,:-1,:]
        flux_diff_x = fluxes_right[:,:,:,:,1:] - fluxes_right[:,:,:,:,:-1]

        return flux_diff_y, flux_diff_x