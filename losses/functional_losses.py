import torch
import math
import torch.nn as nn

def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    """central_diff_3d computes derivatives 
    df(x,y,z)/dx and df(x,y,z)/dy for f(x,y,z) defined 
    on a regular 2d grid using finite-difference

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j,k] = f(x_i, y_j,z_k)
    h : float or list
        discretization size of grid for each dimension
    fix_x_bnd : bool, optional
        whether to fix dx on the x boundaries, by default False
    fix_y_bnd : bool, optional
        whether to fix dy on the y boundaries, by default False
    fix_z_bnd : bool, optional
        whether to fix dz on the z boundaries, by default False

    Returns
    -------
    dx, dy, dz
        tuple such that dx[:, i,j,k]= df(x_i,y_j,z_k)/dx
        and dy[:, i,j,k]= df(x_i,y_j,z_k)/dy
        and dz[:, i,j,k]= df(x_i,y_j,z_k)/dz
    """
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


class H1Loss(object):
    def __init__(self, d=1, measure=1., reduction='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"H1_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        dict_x = {}
        dict_y = {}
        
        if self.d == 3:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature
    
    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff**0.5)/(ynorm**0.5)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        return self.rel(y_pred, y, quadrature=quadrature)


class LpLoss(object):
    def __init__(self, d=1, p=2, measure=1., reduction='sum'):
        super().__init__()

        self.d = d
        self.p = p
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x

    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class HdivLoss(object):
    def __init__(self, d=1, measure=1., reduction='sum', eps=1e-8, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        
        self.eps = eps
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"Hdiv_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        dict_x = {}
        dict_y = {}
        
        if self.d == 3:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            div_x = torch.flatten(x_x + x_y + x_z, start_dim=-3)
            div_y = torch.flatten(y_x + y_y + y_z, start_dim=-3)
        else:
            raise NotImplementedError("HdivLoss is only implemented for 3D at the moment.")
        
        dict_x[1] = div_x
        dict_y[1] = div_y
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature
    
    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2  #compute L2 norm of x-y

        diff += const*torch.norm(dict_x[1] - dict_y[1], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        diff += torch.norm(dict_x[1] - dict_y[1], p=2, dim=-1, keepdim=False) ** 2
        ynorm += torch.norm(dict_y[1], p=2, dim=-1, keepdim=False) ** 2

        diff = (diff**0.5)/(ynorm**0.5 + self.eps)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        return self.rel(y_pred, y, quadrature=quadrature)


def calculate_density(P_m, T_m, P_f, T_f, compressibility, therm_expansion_coeff, P_min, P_max, T_min, T_max):
    P_ref = 100  # kPa
    T_ref = 25 # Celsius
    
    # Un-normalizing pressures and temperatures
    pressure_formation = P_m * (P_max - P_min) + P_min
    temp_formation = T_m * (T_max - T_min) + T_min
    pressure_frac = P_f * (P_max - P_min) + P_min
    temp_frac = T_f * (T_max - T_min) + T_min
    
    # Calculating densities using exponential model
    density_pred_m = 1000.11 * torch.exp(-therm_expansion_coeff * (temp_formation - T_ref) + compressibility * (pressure_formation - P_ref))
    density_pred_f = 1000.11 * torch.exp(-therm_expansion_coeff * (temp_frac - T_ref) + compressibility * (pressure_frac - P_ref))
    
    return density_pred_m, density_pred_f


def get_density_loss(density_pred_m, density_formation, density_pred_f, density_frac,
                     density_min = 850.0, density_max = 1050.0, loss_fn=[H1Loss(d=3, reduction='mean',
                                                                                fix_x_bnd=True,
                                                                                fix_y_bnd=True,
                                                                                fix_z_bnd=True).abs, nn.MSELoss()]):
    
    normalized_rho_pred_m = (density_pred_m - density_min) / (density_max - density_min)
    normalized_rho_m = (density_formation - density_min) / (density_max - density_min)
    normalized_rho_pred_f = (density_pred_f - density_min) / (density_max - density_min)
    normalized_rho_f = (density_frac - density_min) / (density_max - density_min)
    
    loss_m = 0.0
    loss_f = 0.0
    for i in range(len(loss_fn)):
        loss_m += loss_fn[i](normalized_rho_pred_m, normalized_rho_m)
        loss_f += loss_fn[i](normalized_rho_pred_f, normalized_rho_f)
    
    return loss_m, loss_f

def get_material_balance_loss(density_pred_m, density_pred_f,  # DEPRECATED: use MBE loss in ARFNO_main.py instead
                              normalized_rates,
                              phi_m=0.2, phi_frac=0.0004, 
                              V=10 * 10 * 10,
                              rate_scale_factor=5000.0,
                              characteristic_mass=751588.88):
        
    d_rho_m = density_pred_m[:, 1:] - density_pred_m[:, :-1]
    d_rho_f = density_pred_f[:, 1:] - density_pred_f[:, :-1]
    
    accum_m = torch.sum(d_rho_m, dim=(2,3,4)) * phi_m * V
    accum_f = torch.sum(d_rho_f, dim=(2,3,4)) * phi_frac * V
    
    accum_m = accum_m / characteristic_mass
    accum_f = accum_f / characteristic_mass
    
    unscaled_rates = normalized_rates * rate_scale_factor
    unscaled_rates = unscaled_rates * 1000  # from m3/day to kg/day SC
    unscaled_rates = unscaled_rates / characteristic_mass  # normalized rates
    unscaled_rates = torch.sum(unscaled_rates, dim=2)
    
    # material balance
    mb = accum_m + accum_f - unscaled_rates
    mb_loss = torch.mean(mb ** 2)
    return mb_loss
    
    
    
    
    
    
    