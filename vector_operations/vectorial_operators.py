import numpy as np
from einops import rearrange, einsum


def compute_curl(field_nodes, mesh, list_modes = None):
    """Compute the curl of a vector field INITIALLY ON NODES.
    Requirements: 
        numpy, einops
    Args:
        field_nodes[mesh.nn, 6, list_modes.size()]
        mesh
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_gauss[n_gauss, 6, list_modes.size()]: curl of field_nodes evaluated on Gauss points.
    """
    if list_modes is None:
        list_modes = np.arange(field_nodes.shape[-1])

    MF = len(list_modes)
    
    rot_gauss = np.empty((mesh.R_G.shape[0], mesh.R_G.shape[1], 6, field_nodes.shape[-1])) #l_G, me, 6, mF with 6 corresponding to the 6 types for vector

    list_modes = list_modes.reshape(1, 1, MF) #1 1 mF
    rays = mesh.R_G.reshape(mesh.l_G, mesh.me, 1) #l_G me 1
    # mesh.jj with shape nw me and has values in [0, nn-1]
    
    rot_gauss[:, :, 0, :] = list_modes/rays*einsum(field_nodes[mesh.jj, 5, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF') - einsum(field_nodes[mesh.jj, 2, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    rot_gauss[:, :, 2, :] = einsum(field_nodes[mesh.jj, 0, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF') - einsum(field_nodes[mesh.jj, 4, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    rot_gauss[:, :, 4, :] = 1/rays*einsum(field_nodes[mesh.jj, 2, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')+einsum(field_nodes[mesh.jj, 2, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF') - list_modes/rays*einsum(field_nodes[mesh.jj, 1, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')

    rot_gauss[:, :, 1, :] = -list_modes/rays*einsum(field_nodes[mesh.jj, 4, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF') - einsum(field_nodes[mesh.jj, 3, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    rot_gauss[:, :, 3, :] = einsum(field_nodes[mesh.jj, 1, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF') - einsum(field_nodes[mesh.jj, 5, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    rot_gauss[:, :, 5, :] = 1/rays*einsum(field_nodes[mesh.jj, 3, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')+einsum(field_nodes[mesh.jj, 3, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF') + list_modes/rays*einsum(field_nodes[mesh.jj, 0, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')

    return rearrange(rot_gauss, "l_G me c MF -> (l_G me) c MF")





def compute_grad(field_nodes, mesh, list_modes = None):
    """Compute the gradient of a scalar field INITIALLY ON NODES.
    Requirements: 
        numpy, einops
    Args:
        field_nodes[mesh.nn, 2, list_modes.size()]
        mesh
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_gauss[n_gauss, 6, list_modes.size()]: curl of field_nodes evaluated on Gauss points.
    """
    if list_modes is None:
        list_modes = np.arange(field_nodes.shape[-1])

    MF = len(list_modes)
    
    grad_gauss = np.empty((mesh.R_G.shape[0], mesh.R_G.shape[1], 6, field_nodes.shape[-1])) #l_G, me, 6, mF with 6 corresponding to the 6 types for vector

    list_modes = list_modes.reshape(1, 1, MF) #1 1 mF
    rays = mesh.R_G.reshape(mesh.l_G, mesh.me, 1) #l_G me 1
    # mesh.jj with shape nw me and has values in [0, nn-1]
    
    grad_gauss[:, :, 0, :] = einsum(field_nodes[mesh.jj, 0, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    grad_gauss[:, :, 1, :] = einsum(field_nodes[mesh.jj, 1, :], mesh.dw[0, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')

    grad_gauss[:, :, 2, :] = list_modes/rays*einsum(field_nodes[mesh.jj, 1, :], mesh.ww, 'nw me mF, nw l_G me -> l_G me mF')
    grad_gauss[:, :, 3, :] = -list_modes/rays*einsum(field_nodes[mesh.jj, 0, :], mesh.ww, 'nw me mF, nw l_G me -> l_G me mF')

    grad_gauss[:, :, 4, :] = einsum(field_nodes[mesh.jj, 0, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    grad_gauss[:, :, 5, :] = einsum(field_nodes[mesh.jj, 1, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')

    return rearrange(grad_gauss, "l_G me c MF -> (l_G me) c MF")