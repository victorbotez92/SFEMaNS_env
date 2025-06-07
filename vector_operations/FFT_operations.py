import numpy as np
from FFT_IFFT import phys_to_fourier, fourier_to_phys

def FFT_CROSS_PROD(field_1, field_2, list_modes=None):
    """Compute the vectorial product between two fields (can be either on nodes or Gauss points) imported in Fourier space.
    Requirements: 
        numpy
    Args:
        field_1, field_2[X, 6, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_out[X, 6, list_modes.size()]: vectorial product kept on nodes or Gauss points.
    """
    field_1_phys = fourier_to_phys(field_1, list_modes=list_modes)
    field_2_phys = fourier_to_phys(field_2, list_modes=list_modes)
    field_prod_phys = np.zeros(field_1_phys.shape)

    field_prod_phys[:, 0, :] = field_1_phys[:, 1, :]*field_2_phys[:, 2, :] - field_1_phys[:, 2, :]*field_2_phys[:, 1, :]
    field_prod_phys[:, 1, :] = field_1_phys[:, 2, :]*field_2_phys[:, 0, :] - field_1_phys[:, 0, :]*field_2_phys[:, 2, :]
    field_prod_phys[:, 2, :] = field_1_phys[:, 0, :]*field_2_phys[:, 1, :] - field_1_phys[:, 1, :]*field_2_phys[:, 0, :]

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod


def FFT_DOT_PROD(field_1, field_2, list_modes=None):
    """Compute the scalar product between two fields (can be either on nodes or Gauss points) imported in Fourier space.
    Requirements: 
        numpy
    Args:
        field_1, field_2[X, 6, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_out[X, 2, list_modes.size()]: scalar product kept on nodes or Gauss points.
    """
    field_1_phys = fourier_to_phys(field_1, list_modes=list_modes)
    field_2_phys = fourier_to_phys(field_2, list_modes=list_modes)
    field_prod_phys = np.zeros((field_1_phys.shape[0], 1, field_1_phys.shape[2]))

    field_prod_phys[:, 0, :] = np.sum(field_1_phys[:, :, :]*field_2_phys[:, :, :],axis=1)

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod

def FFT_SCAL_VECT_PROD(field_1, field_2, list_modes=None, exponent=1):
    """Compute the product between a scalar and a vector field (can be either on nodes or Gauss points) imported in Fourier space.
    Requirements: 
        numpy
    Args:
        field_1[X, 2, list_modes.size()]
        field_2[X, 6, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
        exponent (optional): 1 if field_1*field_2, -1 if 1/field_1*field_2
    Returns:
        field_out[X, 6, list_modes.size()]: scalar product kept on nodes or Gauss points.
    """
    field_1_phys = fourier_to_phys(field_1, list_modes=list_modes)
    field_2_phys = fourier_to_phys(field_2, list_modes=list_modes)
    field_prod_phys = np.zeros((field_1_phys.shape[0], 3, field_1_phys.shape[2]))

    field_prod_phys[:, :, :] = (field_1_phys[:, :, :])**exponent*field_2_phys[:, :, :]

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod

def FFT_EUCLIDIAN_PROD(field_1, field_2, mesh):
    """Compute the euclidian product between two fields on Gauss points (i.e scalar product and then volume integral without normalization)

    Requirements: 
        numpy
    Args:
        field_1, field_2[n_gauss, 2*D, list_modes.size()]
        mesh
    Returns:
        integral over mesh of field_1.field_2: scalar product kept on nodes or Gauss points.
    """
    field_product = np.zeros(field_1.shape)
    field_product = (field_1*field_2).sum(axis=1)
    field_product[:, 1:] *= 1/2
    field_product = field_product.sum(axis=1)

    return np.sum(field_product*mesh.W.reshape(mesh.l_G*mesh.me))