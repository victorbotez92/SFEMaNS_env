import numpy as np


#=============================================
#=============================================
#==== FFT AND IFFT ===========================
#=============================================
#=============================================



def fourier_to_phys(field_in, mF_max=None): #requires "fourier" format (N, 2*D, MF), outputs "phys" format (N, D, 2*mF_max-1)
    """IFFT implemented with np.fft.ifft(norm='forward')
    Requirements: 
        numpy, einops
    Args:
        field_in, field_2[X, 2*D, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
        mF_max (optional): if none specified, by default set to list_modes[-1]
    Returns:
        field_out[X, D, np.arange(2*MF-1)]: IFFT kept on nodes or Gauss points.
    """
    list_modes = np.arange(field_in.shape[-1])
    mF_max = list_modes[-1]+1

    nb_mF = len(list_modes)
    if nb_mF != field_in.shape[-1]:
        raise IndexError("the list_modes array does not match the amount of Fourier modes within field_in")
    
    N = field_in.shape[0]
    D = field_in.shape[1]//2

    field_in_c = np.zeros((N, D, 2*mF_max - 1), dtype=np.complex128)
    field_in_c[:, :, 0] = field_in[:, ::2, 0]
    field_in_c[:, :, 1:mF_max] = 1/2*(field_in[:, ::2, 1:] - 1.j*field_in[:, 1::2, 1:])
    field_in_c[:, :, mF_max:] = np.conjugate(field_in_c[:, :, 1:mF_max])[:, :, ::-1]


    field_out = np.fft.ifft(field_in_c, axis=2, norm='forward').real
    return field_out #(N, D, 2*mF_max-1)


def phys_to_fourier(field_in): #requires "phys" format (N, D, 2*mF_max-1), outputs "fourier" format (N, 2*D, MF)
    """FFT implemented with np.fft.fft(norm='forward')
    Requirements: 
        numpy, einops
    Args:
        field_in, field_2[X, D, np.arange(2*MF-1)]
    Returns:
        field_out[X, 2*D, MF]: FFT kept on nodes or Gauss points.
    """

    MF = (field_in.shape[-1]+1)//2    

    N = field_in.shape[0]
    D = field_in.shape[1]
    field_out = np.zeros((N, 2*D, MF))

    field_out_c = np.fft.fft(field_in, axis=2, norm='forward')

    field_out[:, ::2, :] = field_out_c[:, :, :MF].real
    field_out[:, 1::2, :] = -field_out_c[:, :, :MF].imag

    field_out[:, :, 1:] *= 2

    return field_out #(N, 2*D, MF)



#=============================================
#=============================================
#=== VECTOR PRODUCTS WITH/WITHOUT PADDING ====
#=============================================
#=============================================



def FFT_CROSS_PROD(field_1, field_2, list_modes=None, pad = True):
    """Compute the vectorial product between two fields (can be either on nodes or Gauss points) imported in Fourier space.
    Requirements: 
        numpy
    Args:
        field_1, field_2[X, 6, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_out[X, 6, list_modes.size()]: vectorial product kept on nodes or Gauss points.
    """
    if list_modes is None:
        assert field_1.shape == field_2.shape
        mF_max = field_1.shape[-1]
        list_modes = np.arange(mF_max)
    else:
        mF_max = list_modes[-1] + 1

    if pad:
        mF_max = int(3.0/2.0*mF_max)
    field_1_pad = np.zeros((field_1.shape[0], field_1.shape[1], mF_max))
    field_1_pad[:, :, np.array(list_modes)] = field_1
    field_1_phys = fourier_to_phys(field_1_pad)
    del field_1_pad

    field_2_pad = np.zeros((field_2.shape[0], field_2.shape[1], mF_max))
    field_2_pad[:, :, np.array(list_modes)] = field_2
    field_2_phys = fourier_to_phys(field_2_pad)
    del field_2_pad

    field_prod_phys = np.zeros(field_2_phys.shape)

    field_prod_phys[:, 0, :] = field_1_phys[:, 1, :]*field_2_phys[:, 2, :] - field_1_phys[:, 2, :]*field_2_phys[:, 1, :]
    field_prod_phys[:, 1, :] = field_1_phys[:, 2, :]*field_2_phys[:, 0, :] - field_1_phys[:, 0, :]*field_2_phys[:, 2, :]
    field_prod_phys[:, 2, :] = field_1_phys[:, 0, :]*field_2_phys[:, 1, :] - field_1_phys[:, 1, :]*field_2_phys[:, 0, :]

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod[:, :, np.array(list_modes)]


def FFT_DOT_PROD(field_1, field_2, list_modes=None, pad = True):
    """Compute the scalar product between two fields (can be either on nodes or Gauss points) imported in Fourier space.
    Requirements: 
        numpy
    Args:
        field_1, field_2[X, 6, list_modes.size()]
        list_modes (optional): if none specified, by default set to np.arange(field_nodes.shape[-1])
    Returns:
        field_out[X, 2, list_modes.size()]: scalar product kept on nodes or Gauss points.
    """
    if list_modes is None:
        assert field_1.shape == field_2.shape
        mF_max = field_1.shape[-1]
        list_modes = np.arange(mF_max)
    else:
        mF_max = list_modes[-1] + 1

    if pad:
        mF_max = int(3.0/2.0*mF_max)
    field_1_pad = np.zeros((field_1.shape[0], field_1.shape[1], mF_max))
    field_1_pad[:, :, np.array(list_modes)] = field_1
    field_1_phys = fourier_to_phys(field_1_pad)
    del field_1_pad

    field_2_pad = np.zeros((field_2.shape[0], field_2.shape[1], mF_max))
    field_2_pad[:, :, np.array(list_modes)] = field_2
    field_2_phys = fourier_to_phys(field_2_pad)
    del field_2_pad

    field_prod_phys = np.zeros((field_1_phys.shape[0], 1, field_1_phys.shape[2]))

    field_prod_phys[:, 0, :] = np.sum(field_1_phys[:, :, :]*field_2_phys[:, :, :],axis=1)

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod[:, :, np.array(list_modes)]

def FFT_SCAL_VECT_PROD(field_1, field_2, list_modes=None, pad = True, exponent=1):
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
    if list_modes is None:
        assert field_1.shape[-1] == field_2.shape[-1] and field_1.shape[0] == field_2.shape[0]
        mF_max = field_1.shape[-1]
        list_modes = np.arange(mF_max)
    else:
        mF_max = list_modes[-1] + 1

    if pad:
        mF_max = int(3.0/2.0*mF_max)
    field_1_pad = np.zeros((field_1.shape[0], field_1.shape[1], mF_max))
    field_1_pad[:, :, np.array(list_modes)] = field_1
    field_1_phys = fourier_to_phys(field_1_pad)
    del field_1_pad

    field_2_pad = np.zeros((field_2.shape[0], field_2.shape[1], mF_max))
    field_2_pad[:, :, np.array(list_modes)] = field_2
    field_2_phys = fourier_to_phys(field_2_pad)
    del field_2_pad

    field_prod_phys = np.zeros((field_1_phys.shape[0], 3, field_1_phys.shape[2]))

    field_prod_phys[:, :, :] = (field_1_phys[:, :, :])**exponent*field_2_phys[:, :, :]

    field_prod = phys_to_fourier(field_prod_phys)
    return field_prod[:, :, np.array(list_modes)]

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