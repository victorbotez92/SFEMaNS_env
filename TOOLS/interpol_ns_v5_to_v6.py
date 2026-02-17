"""
When running this module, the user needs to give two paths as arguments: 
    path_v5p4 containing mesh binaries related to the v5p4 suites => generate these with the application bins_to_mesh
    path_v6 containing mesh binaries related to the v5p4 suites => generate these using save_snapshot from SFEMaNS V6
    I => iteration of ns suite
    check_plots => True or False to check if interpolation went correctly
"""

#================================
#============ IMPORTS ===========
#================================

import numpy as np
import matplotlib.pyplot as plt

import sys, os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path+'/../..')

from SFEMaNS_env import *
cmap = return_cmap()


#================================================
#============ check plots function ==============
#================================================

def make_plot(field, path_out, mesh, title):
    print('plotting ', title)
    D = field.shape[1]//2
    fig, ax = plt.subplots(D, 2, figsize=(10, 20*D/3))
    for a in range(2):
        for d in range(D):
            if D == 1:
                im = ax[a].tripcolor(mesh.tri, field[:, 2*d + a, 0], cmap = cmap)
                fig.colorbar(im, ax=ax[a], orientation='vertical')
            else:
                im = ax[d, a].tripcolor(mesh.tri, field[:, 2*d + a, 0], cmap = cmap)
                fig.colorbar(im, ax=ax[d, a], orientation='vertical')
    fig.tight_layout()
    plt.savefig(path_out+'/'+title+'.png')
    plt.close()

#================================
#============ init ==============
#================================

type_inputs = False

if type_inputs:
    path_v5p4 = "/people/botez/post_pro_sfemans/interpol_v5_to_v6/mesh_v5p4/"
    path_v6 = "/people/botez/post_pro_sfemans/interpol_v5_to_v6/mesh_v6/"
    I = 10
    check_plots = True

else:
    inputs = sys.argv
    if len(inputs) == 1:
        path_v5p4 = input('Path containing mesh v5p4 and suites to be interpolated:   ')
        path_v6 = input('Path containing mesh v6 and where suites are to be created:   ')
        I = int(input('suite iteration number: '))
        check_plots = 1==int(input('Should we check with 2D plots? (0/1)'))

    elif len(inputs) == 4:
        path_v5p4 = inputs[1]
        path_v6 = inputs[2]
        I = int(inputs[3])
        check_plots = False
    
    elif len(inputs) == 5:
        path_v5p4 = inputs[1]
        path_v6 = inputs[2]
        I = int(inputs[3])
        check_plots = True
    else:
        raise IndexError(f'BUG In interpol_v5_to_v6: either do not put arguments, or do put exactly three or four => path_v5p4, path_v6, I, check_plots(optional bool). Currently it is {inputs}')

dict_field = {}
list_char_u = ['un', 'un_m1']
list_char_p = ['pn', 'pn_m1', 'incpn', 'incpn_m1']

mesh_in_vv = define_mesh(path_v5p4, 'vv')
mesh_in_pp = define_mesh(path_v5p4, 'pp')
sfem_in_vv = SFEMaNS_par(path_v5p4, 'vv')
sfem_in_pp = SFEMaNS_par(path_v5p4, 'pp')

mesh_out_vv = define_mesh(path_v6 + "/vvmesh", 'vv')
sfem_out_vv = SFEMaNS_par(path_v6 + "/vvmesh", 'vv')
mesh_out_pp = define_mesh(path_v6 + "/ppmesh", 'pp')
sfem_out_pp = SFEMaNS_par(path_v6 + "/ppmesh", 'pp')

assert mesh_in_vv.R.shape == mesh_out_vv.R.shape, ((mesh_in_vv.R.shape, mesh_out_vv.R.shape))
assert mesh_in_pp.R.shape == mesh_out_pp.R.shape, ((mesh_in_pp.R.shape, mesh_out_pp.R.shape))

sort_in_vv = np.argsort(mesh_in_vv.R**2+(mesh_in_vv.Z+np.pi)**3)
sort_out_vv = np.argsort(mesh_out_vv.R**2+(mesh_out_vv.Z+np.pi)**3)

sort_in_pp = np.argsort(mesh_in_pp.R**2+(mesh_in_pp.Z+np.pi)**3)
sort_out_pp = np.argsort(mesh_out_pp.R**2+(mesh_out_pp.Z+np.pi)**3)


#================================
#============ doing u ===========
#================================

for char_u in list_char_u:
    sfem_in_vv.add_suite_ns(path_v5p4, field=char_u, D=3, replace=True)
    print(f'interpolating {char_u}')
    time, u_in = get_suite(sfem_in_vv, I=I, opt_time = True)
    u_in = u_in[0, :, :, :]
    
    dict_field[char_u] = np.empty(u_in.shape)
    dict_field[char_u][sort_out_vv, :, :] = u_in[sort_in_vv, :, :]
    
    if check_plots:
        make_plot(u_in, path_v5p4, mesh_in_vv, char_u)
    

#================================
#============ doing p ===========
#================================

for char_p in list_char_p:
    sfem_in_pp.add_suite_ns(path_v5p4, field=char_p, D=1, replace=True)
    print(f'interpolating {char_p}')
    time, p_in = get_suite(sfem_in_pp, I=I, opt_time = True)
    p_in = p_in[0, :, :, :]
    
    dict_field[char_p] = np.empty(p_in.shape)
    dict_field[char_p][sort_out_pp, :, :] = p_in[sort_in_pp, :, :]
    
    if check_plots:
        make_plot(p_in, path_v5p4, mesh_in_pp, char_p)

#==================================
#============ Writing suite =======
#==================================

write_suite_ns(sfem_out_vv, path_v6, mesh_out_vv, I=I, un=dict_field['un'], un_m1=dict_field['un_m1'],
               pn=dict_field['pn'], pn_m1=dict_field['pn_m1'], incpn=dict_field['incpn'], incpn_m1=dict_field['incpn_m1'], 
               opt_time = time)


#========================================
#========== Final Checking plots ========
#========================================

if check_plots:
    print("Final plots checking")
    for char_u in list_char_u:
        sfem_out_vv.add_suite_ns(path_v6, field=char_u, D=3, replace=True)
        time, u_in = get_suite(sfem_out_vv, I=I, opt_time = True)
        u_in = u_in[0, :, :, :]
        make_plot(u_in, path_v6, mesh_out_vv, char_u)
        
    for char_p in list_char_p:
        sfem_out_pp.add_suite_ns(path_v6, field=char_p, D=1, replace=True)
        time, p_in = get_suite(sfem_out_pp, I=I, opt_time = True)
        p_in = p_in[0, :, :, :]
        make_plot(p_in, path_v6, mesh_out_pp, char_p)
