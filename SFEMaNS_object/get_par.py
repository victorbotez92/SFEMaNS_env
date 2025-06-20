import numpy as np
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

class SFEMaNS_par:
    def __init__(self, path_to_mesh, mesh_type=None):

    #===============Making sure meshes exist

        if mesh_type is None:
            mesh_type = input("Choose mesh_type in vv, pp, H, phi")

        directory = Path(path_to_mesh)
        search_string = f"{mesh_type}rr_"

        matching_files = [file.name for file in directory.iterdir() if search_string in file.name]
        
        raw_list_meshes = [f".{elm.split('.',1)[1]}" for elm in matching_files]
        list_meshes = list(set(raw_list_meshes))
        nb_meshes = len(list_meshes)
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

        self.bins = False
        self.suites = False
        self.S = S
        self.path_to_mesh = path_to_mesh
        self.mesh_ext = mesh_ext
        self.mesh_type = mesh_type

    def add_suite_ns(self, path_suites, name_suites="suite_ns_", field=None, D=None):
        if self.bins == True:
            raise Exception("your SFEMaNS parameter does already contain information about binaries, please create a new one for suites")
        if self.suites == True:
            print("WARNING : previous data about suites replaced")
        if not self.mesh_type in ["pp", "vv"]:
            raise TypeError(f"SFEMaNS object is on {self.mesh_type} while ns-like suite requires pp or vv")
        if field is None:
            if self.mesh_type == "vv":
                field = input(f"on {self.mesh_type}, possibilities are u, um1")
                if not field in ["u", "um1"]:
                    raise NameError(f"Choose your field in ['u', 'um1']")
                d = 3
            elif self.mesh_type == "pp":
                field = input(f"on {self.mesh_type}, possibilities are p, pm1")
                if not field in ["p", "pm1"]:
                    raise NameError(f"Choose your field in ['p', 'pm1']")
                d = 1

            if (not D is None) and d != D:
                raise ValueError(f"input D={D} differs from D={d} from {field}")
            
        search_string_1 = f"{name_suites}S000_I"
        matching_files = [file.name for file in Path(f"{path_suites}/").iterdir() if ((search_string_1 in file.name) and (self.mesh_ext in file.name))]
        matching_files = [elm.split(search_string_1)[1].split(self.mesh_ext)[0] for elm in matching_files]
        matching_files = list(set(matching_files))
        tab_I = np.sort(np.array(matching_files))
        
        self.suites = True
        self.type_suite = "ns"
        self.path_suites = path_suites
        self.name_suites = name_suites
        self.field = field
        self.tab_I = tab_I

        # include_suite(sfem_par,path_suites,name_suites, field, tab_I)
        # return sfem_par

    def add_suite_maxwell(self, path_suites, name_suites="suite_maxwell_", field=None, D=None):
        if self.bins == True:
            raise Exception("your SFEMaNS parameter does already contain information about binaries, please create a new one for suites")
        if self.suites == True:
            print("WARNING : previous data about suites replaced")
        if not self.mesh_type in ["phi", "H"]:
            raise TypeError(f"SFEMaNS object is on {self.mesh_type} while maxwell-like suite requires phi or H")
        if field is None:
            if self.mesh_type == "H":
                field = input(f"on {self.mesh_type}, possibilities are B, Bm1, H, Hm1")
                if not field in ["B", "Bm1", "H", "Hm1"]:
                    raise NameError(f"Choose your field in ['B', 'Bm1', 'H', 'Hm1]")
                d = 3
            elif self.mesh_type == "phi":
                field = input(f"on {self.mesh_type}, possibilities are phi, phim1")
                if not field in ["phi", "phim1"]:
                    raise NameError(f"Choose your field in ['phi', 'phim1']")
                d = 1

            if (not D is None) and d != D:
                raise ValueError(f"input D={D} differs from D={d} from {field}")
            
        search_string_1 = f"{name_suites}S000_I"
        matching_files = [file.name for file in Path(f"{path_suites}/").iterdir() if ((search_string_1 in file.name) and (self.mesh_ext in file.name))]
        matching_files = [elm.split(search_string_1)[1].split(self.mesh_ext)[0] for elm in matching_files]
        matching_files = list(set(matching_files))
        tab_I = np.sort(np.array(matching_files))
        
        self.suites = True
        self.type_suite = "maxwell"
        self.path_suites = path_suites
        self.name_suites = name_suites
        self.field = field
        self.tab_I = tab_I

        # include_suite(sfem_par,path_suites,name_suites, field, tab_I)
        # return sfem_par

    def add_bins(self, path_binaries, field=None, D=None):
        if self.suites == True:
            raise Exception("your SFEMaNS parameter does already contain information about suites, please create a new one for binaries")
        if self.bins == True:
            print("WARNING : previous data about binaries replaced")
        if field is None:
            list_possible_fields = []

            search_string = "phys"
            phys = len([file.name for file in Path(f"{path_binaries}/").iterdir() if (search_string in file.name)])>0
            if phys:
                search_string_1 = "phys_"
                search_string_2 = "_S0000"
                matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]

                matching_files = [elm.split(search_string_1)[1].split(search_string_2)[0] for elm in matching_files]
                matching_files = list(set(matching_files))
                list_possible_fields = []

                for i,elm in enumerate(matching_files):
                    
                    len_char = len(elm)
                    test_field = elm[:len_char-1]
                    if (f"{test_field}1" in matching_files) and (f"{test_field}2" in matching_files) and (f"{test_field}3" in matching_files):
                        list_possible_fields.append((elm[:len_char-1], 3))
                    else:
                        list_possible_fields.append((elm, 1))

                list_possible_fields = list(set(list_possible_fields)) 

            search_string_1 = "fourier_"
            search_string_2 = "c_S0000_F0000"
            fourier_per_mode = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))])>0
            if fourier_per_mode:
                matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
                matching_files = [elm.split(search_string_1)[1].split(search_string_2)[0] for elm in matching_files]
                matching_files = list(set(matching_files))
                for i,elm in enumerate(matching_files):
                    if elm[-1] == '1' and i+2<len(matching_files) and matching_files[i+1][-1] == '2' and matching_files[i+2][-1] == '3':
                        list_possible_fields.append((elm[:-1], 3))
                    elif elm[-1] == '2' and i+1<len(matching_files) and matching_files[i+1][-1] == '3' and (i-1 >= 0) and matching_files[i-1][-1] == '1':
                        continue
                    elif elm[-1] == '3' and (i-2 >= 0) and matching_files[i-1][-1] == '2' and matching_files[i-2][-1] == '1':
                        continue
                    else:
                        list_possible_fields.append((elm, 1))
                list_possible_fields = list(set(list_possible_fields)) 
                
            search_string_1 = "fourier_"
            search_string_2 = "c_S0000_I"
            fourier = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))])>0
            if fourier:
                matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
                matching_files = [elm.split(search_string_1)[1].split(search_string_2)[0] for elm in matching_files]
                matching_files = list(set(matching_files))
                
                for i,elm in enumerate(matching_files):
                    
                    len_char = len(elm)
                    test_field = elm[:len_char-1]
                    if (f"{test_field}1" in matching_files) and (f"{test_field}2" in matching_files) and (f"{test_field}3" in matching_files):
                        list_possible_fields.append((elm[:len_char-1], 3))
                    else:
                        list_possible_fields.append((elm, 1))

            field = input(f"possible fields and their dimension {list_possible_fields}")
            bool_found_field = False
            i = 0
            while not bool_found_field:
                if list_possible_fields[i][0] == field:
                    bool_found_field = True
                    D = list_possible_fields[i][1]
                else:
                    i += 1

        else:
            if D is None:
                raise TypeError(f"you selected the field {field}, please write its dimension D as well")

        search_string = "phys"
        phys = len([file.name for file in Path(f"{path_binaries}/").iterdir() if (search_string in file.name) and (field in file.name)])>0
        
        search_string_1 = "fourier_"
        search_string_2 = "c_S0000_F0000"
        fourier_per_mode = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (field in file.name) and (self.mesh_ext in file.name))])>0
        
        search_string_1 = "fourier_"
        search_string_2 = "c_S0000_I"
        fourier = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (field in file.name) and (self.mesh_ext in file.name))])>0

        if not (phys or fourier or fourier_per_mode):
            raise NameError(f"{field} was not found in {path_binaries}")

        S = self.S
        tab_I = None
        MF = None

        if phys:
            search_string_1 = f"phys_{field}"
            search_string_2 = "_S0000_I"
            matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
            
            # tab_I = np.sort(np.array([int((elm.split('.',0)).split('I')[-1])]))
            tab_I = [int((elm.split(self.mesh_ext)[0]).split(search_string_2)[-1]) for elm in matching_files]
            tab_I = list(set(tab_I))
            tab_I = np.sort(np.array(tab_I))
            
            N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_binaries}/phys_{field}{char}_S0000_I{tab_I[0]:04d}{self.mesh_ext}'
            mode = np.fromfile(directory)
            MF = ((len(mode)//N)+1)//2            

        if fourier:
            search_string_1 = f"fourier_{field}"
            search_string_2 = "c_S0000_I"
            matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
            
            # tab_I = np.sort(np.array([int((elm.split('.',0)).split('I')[-1])]))
            tab_I = [int((elm.split(self.mesh_ext)[0]).split(search_string_2)[-1]) for elm in matching_files]
            tab_I = list(set(tab_I))
            tab_I = np.sort(np.array(tab_I))
            
            N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_binaries}/fourier_{field}{char}c_S0000_I{tab_I[0]:04d}{self.mesh_ext}'
            mode = np.fromfile(directory)
            if (not MF is None) and MF != len(mode)//N:
                raise ValueError(f"phys file and fourier files do not match (computed MF = {MF} on one hand and MF = {len(mode)//N} on the other hand)")
            else:
                MF = len(mode)//N

        if fourier_per_mode:
            search_string_1 = f"fourier_{field}"
            search_string_2 = "_S0000_F"
            matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]

            if (not MF is None) and MF != len(matching_files)//2//D:
                raise ValueError(f"phys/fourier and fourier_per_mode files do not match (computed MF = {MF} on one hand and MF = {len(matching_files)//2//D} on the other hand)")
            else:
                MF != len(matching_files)//2//D

            N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            if D == 3:
                char = '1'
            elif D == 1:
                char = ''
            directory = f'{path_binaries}/fourier_{field}{char}c_S0000_F0000{self.mesh_ext}'
            mode = np.fromfile(directory)
            nb_I = len(mode)//N
            if (not tab_I is None) and len(tab_I) != nb_I:
                raise ValueError(f"phys/fourier and fourier_per_mode files do not match (found {len(tab_I)} iterations on one hand and {nb_I} iterations on the other hand)")
            else:
                tab_I = 1+np.arange(nb_I)

        self.bins = True
        self.path_bins = path_binaries
        self.phys = phys
        self.fourier_per_mode = fourier_per_mode
        self.fourier = fourier
        self.field = field
        self.I = tab_I
        self.MF = MF
        self.D = D
        # include_binaries(sfem_par,phys,fourier_per_mode,fourier,tab_I,MF)
        # return sfem_par