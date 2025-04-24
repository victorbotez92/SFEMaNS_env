from get_par import *
from read_stb import *
from read_suite import *
from manip_suite import *


def intro_sfemans():
    # print('Welcome to the SFEMaNS environment for python post-processing.')
    # print('=========Getting started=========')
    # print('You should define once and for all the following variables: ')
    # print('S: Number of meridonal sections')
    # print('D: Dimension of the field')
    # print('field: Name your field within the following possibilities: ')
    # print('    u, p, H, B, phi...')
    # print('path_to_mesh: path containing the vvrr, Hrr, etc...')
    # print('mesh_ext: the name of your mesh (should be like .MHD_RECT_80_40_H_2R_1_sym.FEM for instance)')
    # print('MF: the number of Fourier modes you would like to consider (can be smaller than the actual MF used for computations through SFEMaNS)')
    # print()
    # print('   ')
    print('=========Importing data=========')
    # print('Functions to import data from binaries or directly from SFEMaNS suites are defined within get_data.py')
    print()
    print('SFEMaNS_par(path_to_binaries_out)')
    print('get_suite(par,I=[],MF=[],fourier_type=["c","s"],opt_extension='',record_stack_lenght=7, get_gauss_points=True,stack_domains=True)')
    print('#output shape is (I MF D a N)')
    print('')
    print('get_phys(par,I)')
    print('#output shape is (I theta N)')
    print('')
    print('get_fourier(par,I,MF=[],fourier_type=["c","s"]):')
    print('#output shape is (MF D a N)')
    print('')
    print('get_fourier_per_mode(par,mF,T=-1,fourier_type=["c","s"]):')
    print('# output shape is (T D a N)')
    print('')
    print('get_mesh(par):')
    print('output is R,Z,W')