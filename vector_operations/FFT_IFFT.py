import numpy as np

def fourier_to_phys(field_in,list_modes=None, mF_max=None): #requires "fourier" format (N, 2*D, MF), outputs "phys" format (N, D, 2*mF_max-1)
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
    if list_modes is None:
        list_modes = np.arange(field_in.shape[-1])
        print("WARNING: setting list_modes to np.arange by default")

    if mF_max is None:
        mF_max = list_modes[-1]+1
        print("WARNING: setting mF_max to list_modes[-1] by default")

    nb_mF = len(list_modes)
    if nb_mF != field_in.shape[-1]:
        raise IndexError("the list_modes array does not match the amount of Fourier modes within field_in")
    
    N = field_in.shape[0]
    D = field_in.shape[1]//2

    field_in_c = np.zeros((N, D, 2*mF_max - 1), dtype=np.complex128)
    field_in_c[:, :, 0] = field_in[:, ::2, 0]
    field_in_c[:, :, 1:mF_max] = 1/2*(field_in[:, ::2, 1:] - 1.j*field_in[:, 1::2, 1:])
    field_in_c[:, :, mF_max:] = np.conjugate(field_in_c[:, :, 1:mF_max])[:, :, ::-1]

    # angles = 2*np.pi*np.arange(2*mF_max-1)/(2*mF_max-1)
    # angles = angles.reshape(1, 1, 1, 2*mF_max-1)
    # list_modes = list_modes.reshape(1, 1, len(list_modes), 1)

    # field_out = np.sum(np.cos((angles+shift)*list_modes)*(field_in[:, ::2, :].reshape(N, D, nb_mF, 1)),axis=2) + np.sum(np.sin((angles+shift)*list_modes)*(field_in[:, 1::2, :].reshape(N, D, nb_mF, 1)),axis=2)
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

    # field_out[:, ::2, 0] = field_out_c[:, :, 0].real
    # field_out[:, ::2, 1:] = 2*field_out_c[:, :, 1:MF].real
    # field_out[:, 1::2, 1:] = -2*field_out_c[:, :, 1:MF].imag

    # list_modes = np.arange(MF)

    # angles = 2*np.pi*np.arange(2*MF-1)/(2*MF-1)
    # angles = angles.reshape(1, 1, 1, 2*MF-1)
    # list_modes = list_modes.reshape(1, 1, MF, 1)

    # field_out[:, ::2, :] = np.mean(np.cos((angles+shift)*list_modes)*(field_in[:, :, :].reshape(N, D, 1, 2*MF-1)),axis=3) 
    # field_out[:, 1::2, :] = np.mean(np.sin((angles+shift)*list_modes)*(field_in[:, :, :].reshape(N, D, 1, 2*MF-1)),axis=3)
    # field_out *= 2
    # field_out[:, ::2, 0] *= 1/2
    return field_out #(N, 2*D, MF)