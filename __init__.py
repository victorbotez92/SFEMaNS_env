__author__ = "Victor Botez"
__version__ = "0.2.0"



from .SFEMaNS_object import SFEMaNS_par, define_mesh

from .vector_manipulation.operators import gauss_to_nodes, nodes_to_gauss, curl, grad, advection_vect
from .vector_manipulation.FFT_operations import fourier_to_phys, phys_to_fourier
from .vector_manipulation.FFT_operations import FFT_CROSS_PROD, FFT_DOT_PROD, FFT_SCAL_VECT_PROD, FFT_EUCLIDIAN_PROD, SIMPLE_SCAL_VECT_PROD


from .read_write_SFEMaNS.read_suite import get_suite
from .read_write_SFEMaNS.read_stb import get_mesh_gauss, get_phys, get_fourier, get_fourier_per_mode
from .read_write_SFEMaNS.write_stb import write_phys, write_fourier, write_fourier_per_mode

print("help_SFEMaNS()/return_cmap()")

def help_SFEMaNS():
    print('=========Importing/writing data=========')
    print()
    print('SFEMaNS_par, define_mesh')
    print('get_suite, get_phys, get_fourier, get_fourier_per_mode', 'get_mesh_gauss')
    print('write_phys, write_fourier, write_fourier_per_mode')
    print()
    print('WARNING: R, Z, W from get_mesh_gauss ==> shape (me l_G)')
    print()
    print('=========Operators and FFT/IFFT==========')
    print('gauss_to_nodes, nodes_to_gauss, curl, grad, advection_vect')
    print('fourier_to_phys, phys_to_fourier')
    print()
    print('=========Vector products===========')
    print('FFT_CROSS_PROD, FFT_DOT_PROD, FFT_SCAL_VECT_PROD, FFT_EUCLIDIAN_PROD, SIMPLE_SCAL_VECT_PROD')
    print()
    print()
    print('============ TO DO ============')
    print('add divergence, grad for vectors, check read suite, add write suite')

def return_cmap():
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ezmap", ["tab:blue","darkblue","cyan","green",'gold',"darkred","tab:red"])
    return cmap
