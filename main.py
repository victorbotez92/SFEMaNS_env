from get_par import *
from read_stb import *
from read_suite import *
from manip_suite import *


def intro_sfemans():
    print('=========Importing data=========')
    # print('Functions to import data from binaries or directly from SFEMaNS suites are defined within get_data.py')
    print()
    print('SFEMaNS_par(path_to_mesh,opt_path_binaries_out='',opt_path_suite='',field=None)')
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