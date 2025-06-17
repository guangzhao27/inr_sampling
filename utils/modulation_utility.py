import sys
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
from functools import partial
from coral.utils.models.scheduling import ode_scheduling
from torchdiffeq import odeint
def modulation_fix(modulations, n_samples, T, latent_dim, z_transform=None, device=None):
    if device is not None:
        modulations = modulations.to(device)
    modulations = modulations.reshape(n_samples, T, latent_dim)
    modulations = modulations.permute(0, 2, 1)
    if z_transform is not None:
        modulations = z_transform(modulations)
    return modulations

def ode_pred_z(model, graph, modulations, timestamps, epsilon, parameterized):
    if not parameterized:
        _f = model
    else:
        _f = partial(model, graph.pde_parameter)
    
    z_pred = ode_scheduling(odeint, _f, modulations, timestamps, epsilon)
    
    return z_pred