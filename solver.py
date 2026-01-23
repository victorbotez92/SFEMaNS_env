import numpy as np
from einops import rearrange, einsum, repeat

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized


#################
################

def solve_laplace(field_in, mesh, horizontal=False):  #field of shape ((l_G me) D MF)
    """
    solve laplace equation with Neumann BC

        field_in : ((l_G me), D, MF)
        mesh : output from define_mesh deprived of duplicates
        horizontal : if only want to consider horizontal laplacian

        field_out : (N, D, MF) on nodes
    """



    if field_in.shape[0] != mesh.l_G*mesh.me:
        raise TypeError(f"Expected gauss points (i.e {mesh.l_G*mesh.me}), got {field_in.shape[0]} (out of {field_in.shape})")
    if mesh.tab_duplicates is None:
        raise ValueError("The mesh you provided to solve Laplace equation may contain duplicates, first make sure it doesn't")

    MF = field_in.shape[-1]
    field = rearrange(field_in[:, :, :], '(me l_G) D MF-> D l_G me MF', l_G = mesh.l_G)#[np.array([0, 1])]

#==================BUILDING MASS MATRIX



    # Step 1: effective weights (W * detJ) → shape: (lG, me)
    W = einsum(mesh.R[mesh.jj], mesh.ww, mesh.rj, 'nw me, nw l_G, l_G me -> l_G me')
    W = rearrange(W, 'l_G me -> (me l_G)')

    W_eff = rearrange(W, '(me l_G) -> l_G me', l_G=mesh.l_G) #* mesh.rj  # shape: (l_G, me)
    nw = mesh.nw
    # # Step 2: Triplet assembly arrays (mesh.jj is (nw, me))

    R_gauss = einsum(mesh.R[mesh.jj], mesh.ww, 'nw me, nw l_G -> l_G me')

    V = mesh.dw[0, :, :, :].reshape(mesh.nw, 1, mesh.l_G, mesh.me) * mesh.dw[0, :, :, :].reshape(1, mesh.nw, mesh.l_G, mesh.me) * (mesh.rj*R_gauss).reshape(1, 1, mesh.l_G, mesh.me)

    V += (not horizontal) * mesh.dw[1, :, :, :].reshape(mesh.nw, 1, mesh.l_G, mesh.me) * mesh.dw[1, :, :, :].reshape(1, mesh.nw, mesh.l_G, mesh.me) * (mesh.rj*R_gauss).reshape(1, 1, mesh.l_G, mesh.me)
    V_th = mesh.ww.reshape(mesh.nw, 1, mesh.l_G, 1) * mesh.ww.reshape(1, mesh.nw, mesh.l_G, 1) * (mesh.rj/R_gauss).reshape(1, 1, mesh.l_G, mesh.me)
    I = rearrange(repeat(mesh.jj, 'I me -> I J me', J=mesh.jj.shape[0]), 'I J me -> (I J me)') #Final shape (nw * nw * me)
    J = rearrange(repeat(mesh.jj, 'J me -> I J me', I=mesh.jj.shape[0]), 'I J me -> (I J me)')
    V = rearrange(-V.sum(axis=2), 'I J me -> (I J me)')
    V_th = rearrange(-V_th.sum(axis=2), 'I J me -> (I J me)')


    nn = np.max(mesh.jj) + 1  # number of global nodes

    field_out = np.empty((mesh.jj.max()+1, 2, field_in.shape[-1]))

    for mF in range(MF):
        M = coo_matrix((V+mF**2*V_th, (I, J)), shape=(nn, nn)).tocsc()

#============= SOLVING LINEAR SYSTEM
        M_solver = factorized(M)
        n_snap=field.shape[0]#field_nodes.shape[0]

        # Result: (n_snap, nw, me)
        contrib = einsum( mesh.ww, field[:, :, :, mF], W_eff, "nw l_G , n_snap l_G me , l_G me -> n_snap nw me")

        # Initialize B: shape (n_snap, nn)
        B = np.zeros((n_snap, nn))

        # Scatter each node contribution
        for i in range(nw):
            idx = mesh.jj[i, :]  # shape (me,)
            np.add.at(B, (slice(None), idx), contrib[:, i, :])

        field_out[:, :, mF] = np.stack([M_solver(B[k]) for k in range(n_snap)], axis=1)

    return field_out
