import numpy as np
from einops import rearrange, einsum, repeat

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized


def nodes_to_gauss(field, mesh):
    # for s in range(par.S):
        #arange field by triangle
    if len(field.shape) == 2:
        return einsum(field, mesh.ww, 'nw me, nw l_G -> l_G me')
    else:
        return einsum(field, mesh.ww, 'n_snap nw me, nw l_G -> n_snap l_G me')
    

def gauss_to_nodes(field_in, mesh):  #field of shape ((l_G, me), D, MF)
    MF = field_in.shape[-1]
    field = rearrange(field_in, '(me l_G) D MF -> (D MF) l_G me', l_G = mesh.l_G)
    # if len(field.shape) == 2:
    #     field = np.array([field])

#==================BUILDING MASS MATRIX

    # Step 1: effective weights (W * detJ) → shape: (lG, me)
    W_eff = mesh.W * mesh.rj  # shape: (l_G, me)
    nw = mesh.nw
    # Step 2: Triplet assembly arrays (mesh.jj is (nw, me))

    I = rearrange(repeat(mesh.jj, 'I me -> I J me', J=mesh.jj.shape[0]), 'I J me -> (I J me)') #Final shape (nw * nw * me)
    J = rearrange(repeat(mesh.jj, 'J me -> I J me', I=mesh.jj.shape[0]), 'I J me -> (I J me)') 
    V = mesh.ww.reshape(nw, 1, mesh.l_G, 1) * mesh.ww.reshape(1, nw, mesh.l_G, 1) * W_eff.reshape(1, 1, mesh.l_G, mesh.me) #I is phi_i and J is phi_J
    V = rearrange(V.sum(axis=2), 'I J me -> (I J me)')

    nn = np.max(mesh.jj) + 1  # number of global nodes

    M = coo_matrix((V, (I, J)), shape=(nn, nn)).tocsc()

#============= SOLVING LINEAR SYSTEM
    M_solver = factorized(M) 
    n_snap=field.shape[0]#field_nodes.shape[0]

    # Result: (n_snap, nw, me)
    contrib = einsum( mesh.ww, field, W_eff, "nw l_G , n_snap l_G me , l_G me -> n_snap nw me")

    # Initialize B: shape (n_snap, nn)
    B = np.zeros((n_snap, nn))

    # Scatter each node contribution
    for i in range(nw):
        idx = mesh.jj[i, :]  # shape (me,)
        np.add.at(B, (slice(None), idx), contrib[:, i, :])

    U = np.stack([M_solver(B[k]) for k in range(n_snap)], axis=0)
    # U = send_to_triangle(U, mesh)
    return rearrange(U, '(D MF) N -> N D MF', MF = MF)


def send_to_triangle(field, mesh):
    if len(field.shape) == 1:
        field = np.array([field])
        n_snap = 1
        added_axis = True
    else:
        n_snap = field.shape[0]
        added_axis = False
        
    field_rearranged = np.empty((n_snap, mesh.nw, mesh.me))

    field_rearranged = field[:, mesh.jj]
    if added_axis:
        field_rearranged = field_rearranged[0, :, :]
    return field_rearranged


#=============================================
#=============================================
#========= VECTORIAL OPERATORS     ===========
#=============================================
#=============================================



def curl(field_nodes, mesh, list_modes = None):
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





def grad(field_nodes, mesh, list_modes = None):
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

    grad_gauss[:, :, 2, :] = list_modes/rays*einsum(field_nodes[mesh.jj, 1, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')
    grad_gauss[:, :, 3, :] = -list_modes/rays*einsum(field_nodes[mesh.jj, 0, :], mesh.ww, 'nw me mF, nw l_G -> l_G me mF')

    grad_gauss[:, :, 4, :] = einsum(field_nodes[mesh.jj, 0, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')
    grad_gauss[:, :, 5, :] = einsum(field_nodes[mesh.jj, 1, :], mesh.dw[1, :, :, :], 'nw me mF, nw l_G me -> l_G me mF')

    return rearrange(grad_gauss, "l_G me c MF -> (l_G me) c MF")