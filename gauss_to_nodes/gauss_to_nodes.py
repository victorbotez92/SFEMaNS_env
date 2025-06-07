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
    

def gauss_to_nodes(field, mesh):  #field of shape (C, l_G, me)
    if len(field.shape) == 2:
        field = np.array([field])

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

    return send_to_triangle(U, mesh)


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