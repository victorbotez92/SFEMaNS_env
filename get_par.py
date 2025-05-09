import numpy as np

from einops import rearrange,repeat
import os
from pathlib import Path


dict_mesh_type = {
    'u':'vv',
    'p':'pp',
    'B':'H',
    'H':'H',
    'phi':'phi'}

dict_dimension = {
    'u':3,
    'p':1,
    'B':3,
    'H':3,
    'phi':1}

class class_SFEMaNS:
    # def __init__(self,S,D,field,MF,I,I_min,I_freq,path_to_suite,path_to_mesh,mesh_ext,phys,fourier,fourier_per_mode):
    def __init__(self,S,D,field,path_to_suite,path_to_mesh,mesh_ext):

        self.S = S
        self.D = D
        self.field = field
        # self.MF = MF
        # self.I = I
        # self.I_min = I_min
        # self.I_freq = I_freq
        self.path_to_suite = path_to_suite
        self.path_to_mesh = path_to_mesh
        self.mesh_ext = mesh_ext

        self.mesh_type = dict_mesh_type[field]

        # self.phys = phys
        # self.fourier = fourier
        # self.fourier_per_mode = fourier_per_mode

def include_binaries(sfem_par,phys,fourier_per_mode,fourier,I_min,I_freq,I,MF):
        
    sfem_par.phys = phys
    sfem_par.fourier_per_mode = fourier_per_mode
    sfem_par.fourier = fourier
    sfem_par.I_freq = I_freq
    sfem_par.I_min = I_min
    sfem_par.I = I
    sfem_par.MF = MF

    return sfem_par

def include_suite(sfem_par,opt_path_suite,list_extensions,opt_start,opt_I,suite_I_min,suite_I_max):
    
    sfem_par.opt_path_suite = opt_path_suite
    sfem_par.list_extensions = list_extensions
    sfem_par.opt_start = opt_start
    sfem_par.opt_I = opt_I
    sfem_par.suite_I_min = suite_I_min
    sfem_par.suite_I_max = suite_I_max

    return sfem_par

def SFEMaNS_par(path_to_mesh,opt_path_binaries_out='',opt_path_suite='',field=None):


#=============Defining field
    if field is None:
        field = input("Type in the field you're interested in: ")

    try:
        mesh_type = dict_mesh_type[field]
    except NameError:
        raise NameError("The field you typed does not exist, make sure you haven't made a mistake or manually add it to the library")
    if not os.path.exists(path_to_mesh+f'/{field}/') and opt_path_binaries_out=='' and opt_path_suite=='':
        print("WARNING: You consider neither binaries nor suites.")

#============Defining path to binaries
    if opt_path_binaries_out != '':
        no_bins = False
        path_to_suite = opt_path_binaries_out+f'/{field}/'

    else:
        if not os.path.exists(path_to_mesh+f'/{field}/'):
            no_bins = True
        else:
            no_bins = False
        path_to_suite = path_to_mesh+f'/{field}/'

#===============Defining D
    D = dict_dimension[field]

#===============Making sure meshes exist
    mesh_type = dict_mesh_type[field]
    directory = Path(path_to_mesh)
    search_string = f"{mesh_type}rr_"

    matching_files = [file.name for file in directory.iterdir() if search_string in file.name]
    
    raw_list_meshes = [f".{elm.split('.',1)[1]}" for elm in matching_files]
    list_meshes = list(set(raw_list_meshes))
    nb_meshes = len(list_meshes)
    print(f"Found {nb_meshes} different meshes.")
    if nb_meshes == 0:
        raise NameError("The folder you selected does not contain any mesh.")
    elif nb_meshes > 1:
        print(f"The binaries_out file contains several different meshes, type the number between 0 and {nb_meshes-1} corresponding to your mesh")
        num_mesh = int(input())
    else:
        num_mesh = 0

#===============Defining mesh_ext
    mesh_ext = list_meshes[num_mesh]

    search_string = f"{mesh_type}rr_"
    matching_files = [file.name for file in directory.iterdir() if ((search_string in file.name) and (mesh_ext in file.name))]
    list_s = [int((elm.split(".")[0]).split("S")[-1]) for elm in matching_files]

#===============Defining S
    S = max(list_s) + 1

#===============Defining SFEMaNS object
    out = class_SFEMaNS(S,D,field,path_to_suite,path_to_mesh,mesh_ext)

    # if no_bins:
    #     phys = False
    #     fourier_per_mode = False
    #     fourier = False
    #     I_freq = -1
    #     I_min = 0
    #     I = 0
    #     MF = 0
    if not no_bins:
        search_string = "phys"
    #===============Defining phys
        phys = len([file.name for file in Path(f"{path_to_suite}/").iterdir() if (search_string in file.name)])>0

        search_string_1 = "fourier"
        search_string_2 = "S0000_F"
    #===============Defining fourier_per_mode
        fourier_per_mode = len([file.name for file in Path(f"{path_to_suite}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name))])>0

        search_string_1 = "fourier"
        search_string_2 = "S0000_I"
    #===============Defining fourier
        fourier = len([file.name for file in Path(f"{path_to_suite}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name))])>0

    #===============Defining I
        if phys:
            search_string = "phys"
            matching_files = [file.name for file in Path(f"{path_to_suite}/").iterdir() if ((search_string in file.name) and (mesh_ext in file.name))]
            
            # tab_I = np.sort(np.array([int((elm.split('.',0)).split('I')[-1])]))
            tab_I = [int((elm.split('.')[0]).split('I')[-1]) for elm in matching_files]
            tab_I = list(set(tab_I))
            tab_I = np.sort(np.array(tab_I))
            I_min = tab_I[0]
            if len(tab_I)>1:
                I_freq = tab_I[1]-tab_I[0]
            else:
                I_freq = -1
            I = len(matching_files)//S//D


        elif fourier:
            search_string_1 = "fourier"
            search_string_2 = "S0000_I"
            matching_files = [file.name for file in Path(f'{path_to_binaries_out}/').iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (mesh_ext in file.name))]
            tab_I = [int((elm.split('.')[0]).split('I')[-1]) for elm in matching_files]
            tab_I = list(set(tab_I))
            tab_I = np.sort(np.array(tab_I))
            I_min = tab_I[0]
            if len(tab_I)>1:
                I_freq = tab_I[1]-tab_I[0]
            else:
                I_freq = -1
            I = len(matching_files)//D//len(["c","s"])

        elif fourier_per_mode:
            N = len(np.fromfile(path_to_mesh+f'{mesh_type}rr_S0000{mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_to_suite}/fourier_{field}{char}c_S0000_F0000{mesh_ext}'
            mode = np.fromfile(directory)
            I = len(mode)//N
            if not (fourier or phys):
                I_min = 0
                I_freq = -1

    #===============Defining MF
        if fourier_per_mode:
            search_string_1 = "fourier"
            search_string_2 = "S0000_F"
            matching_files = [file.name for file in Path(f'{path_to_suite}/').iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (mesh_ext in file.name))]
            MF = len(matching_files)//D//len(["c","s"])
        
        elif fourier:
            N = len(np.fromfile(path_to_mesh+f'{mesh_type}rr_S0000{mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_to_suite}/fourier_{field}{char}c_S0000_I{I_min:04d}{mesh_ext}'
            mode = np.fromfile(directory)
            MF = len(mode)//N

        elif phys:
            N = len(np.fromfile(path_to_mesh+f'{mesh_type}rr_S0000{mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_to_suite}/phys_{field}{char}_S0000_I{I_min:04d}{mesh_ext}'
            mode = np.fromfile(directory)
            MF = ((len(mode)//N)+1)//2

#==============Updating SFEMaNS object
        out = include_binaries(out,phys,fourier_per_mode,fourier,I_min,I_freq,I,MF)
    # out = class_SFEMaNS(S,D,field,MF,I,I_min,I_freq,path_to_suite,path_to_mesh,mesh_ext,phys,fourier,fourier_per_mode)

#==============Consider opt_path_suite
    if opt_path_suite != '':
        #===== define opt_path_suite
        # out.opt_path_suite = opt_path_suite

        if field == 'u' or field == 'p':
            search_string_1 = 'ns'
        elif field == 'B' or field =='H' or field =='phi':
            search_string_1 = 'maxwell'
        search_string_2 = 'S000'
        matching_files = [file.name for file in Path(opt_path_suite).iterdir() if ((search_string_1 in file.name) and ((search_string_2 in file.name) and (mesh_ext in file.name)))]
        list_extensions = list(set([elm.split(search_string_1)[0] for elm in matching_files]))
        if 'suite_' in list_extensions:
            list_extensions.remove('suite_')
            list_extensions = ['suite_'] + list_extensions
        #===== define suite_extension
        # out.list_extensions = list_extensions

        opt_start = []
        for extension in list_extensions:
            search_string = f'{extension}{search_string_1}_{search_string_2}{mesh_ext}'
            if len([file.name for file in Path(opt_path_suite).iterdir() if (search_string in file.name)]) > 0:
                opt_start.append(True)
            else:
                opt_start.append(False)

        #===== define opt_start
        # out.opt_start = opt_start

        opt_I = []
        suite_I_min = []
        suite_I_max = []
        for extension in list_extensions:
            search_string = f'{extension}{search_string_1}_{search_string_2}_I'
            matching_files = [file.name for file in Path(opt_path_suite).iterdir() if (search_string in file.name)]
            list_I = np.array([int((elm.split(search_string)[1]).split(mesh_ext)[0]) for elm in matching_files])
            opt_I.append(len(list_I)>0)

            if opt_I[-1]:
                suite_I_min.append(min(list_I))
                suite_I_max.append(max(list_I))
            else:
                suite_I_min.append(-1)
                suite_I_max.append(-2)


        #===== define opt_I and suite_I_min/max
        # out.opt_I = opt_I
        # out.suite_I_min = suite_I_min
        # out.suite_I_max = suite_I_max
        out = include_suite(out,opt_path_suite,list_extensions,opt_start,opt_I,suite_I_min,suite_I_max)
    return out