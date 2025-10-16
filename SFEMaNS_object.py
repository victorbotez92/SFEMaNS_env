import numpy as np
from pathlib import Path


def read_mesh_info(path):
    with open(path) as file:
        line = file.readlines()
        values = line[1::2][:4]
        n_w = int(values[0])
        l_G = int(values[1])
        me  = int(values[2])
        n_p = int(values[3])
    return [n_w,l_G,me,n_p]

def read_mesh_info_surface(path):
    with open(path) as file:
        line = file.readlines()
        values = line[11::2][:3]
        n_ws = int(values[0])
        l_Gs = int(values[1])
        mes = int(values[2])
    return [n_ws,l_Gs,mes]

class define_mesh:
    """
    Class to define a mesh object with all its attributes
("path_to_mesh, mesh_ext, S, ME, n_w, l_G")
("On nodes : R, Z")
("jj, ww, dw, rj, nw, lG, me, nn")
opt surface

methods:
    - "rm_duplicate()"
    - "find_duplicate()"
    - "build_tab_sym()"

    """
    def __init__(self, path_to_mesh, mesh_type, mesh_ext = None, surface=False):
        #============== USEFUL STUFF FOR IS NONE
        self.tab_duplicates = None #if rm_duplicate present or not
        #============== USEFUL STUFF FOR IS NONE
        directory = Path(path_to_mesh)
        # search_string = f"{mesh_type}rr_"
        search_string = f"{mesh_type}mesh_rr_node_S0000"
        matching_files = [file.name for file in directory.iterdir() if search_string in file.name]
        if mesh_ext is None:
            raw_list_meshes = [f".{elm.split('.',1)[1]}" for elm in matching_files]
            list_meshes = list(set(raw_list_meshes))
            if len(list_meshes) > 1:
                raise OSError(f"several meshes available, choose one in {list_meshes}")
            elif len(list_meshes) == 0:
                raise OSError(f"No mesh available in {path_to_mesh}")
            else:
                mesh_ext = list_meshes[0]
        else:
            if not mesh_ext in matching_files:
                raise OSError(f"Mesh {mesh_ext} not available in {path_to_mesh}")
            
        self.path_to_mesh = path_to_mesh
        self.mesh_ext = mesh_ext

        search_string_1 = f"{mesh_type}mesh_rr_node_S"
        matching_files = [file.name for file in directory.iterdir() if ((search_string_1 in file.name) and (mesh_ext in file.name))]
        tab_S = [int((elm.split(search_string_1)[1]).split(mesh_ext)[0]) for elm in matching_files]
        S = np.array(tab_S).max() + 1
        self.S = S

        ME = [ read_mesh_info(path_to_mesh+f"/{mesh_type}mesh_info_S{s:04d}.txt")[2] for s in range(S) ]  
        n_w = read_mesh_info(path_to_mesh+f"/{mesh_type}mesh_info_S0000.txt")[0]
        l_G = read_mesh_info(path_to_mesh+f"/{mesh_type}mesh_info_S0000.txt")[1]

        self.ME = ME
        self.n_w = n_w
        self.l_G = l_G
        
        R_node = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}mesh_rr_node_S{s:04d}"+mesh_ext) for s in range(S)])
        Z_node = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}mesh_zz_node_S{s:04d}"+mesh_ext) for s in range(S)])
        self.R = R_node
        self.Z = Z_node
        # self.mesh_type = mesh_type

#============= DOING jj

        mesh_jj = [ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_jj_S{s:04d}"+mesh_ext,dtype=np.int32).reshape(n_w,ME[s],order="F") 
                for s in range(S) ]
        
        nodes_per_S = np.array([mesh_jj[s].max() for s in range(S)])
        cumul_nodes_per_S=np.cumsum(nodes_per_S)
        for s in range(1,S):
            mesh_jj[s]+=cumul_nodes_per_S[s-1]
        mesh_jj = np.hstack(mesh_jj)-1
#============== DOING ww
        mesh_ww = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_ww_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(n_w,l_G,order="F") 
                        for s in range(S) ])
        if S > 1:
            ww_assertion = mesh_ww[:,:l_G] == mesh_ww[:,l_G:2*l_G]
            for s in range(2, S):
                ww_assertion = np.logical_and(ww_assertion, mesh_ww[:,:l_G]==mesh_ww[:,s*l_G:(s+1)*l_G])
            assert np.prod(ww_assertion)
        mesh_ww=mesh_ww[:,:l_G]
#=============== DOING rj
        mesh_rj = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_rj_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(l_G,ME[s],order="F") 
                        for s in range(S) ])
        
        self.jj = mesh_jj
        self.ww = mesh_ww
        self.rj = mesh_rj

        nw, lG, me = mesh_ww.shape[0], mesh_ww.shape[1], mesh_jj.shape[1]
        self.nw, self.lG, self.me = nw, lG, me

        nn = np.max(mesh_jj) + 1
        self.nn = nn

        mesh_dw = np.concatenate([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_dw_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(2,n_w,l_G,ME[s],order="F") 
                        for s in range(S) ], axis=3)
        self.dw = mesh_dw

        if surface:
            MEs = [ read_mesh_info_surface(path_to_mesh+f"/{mesh_type}mesh_info_S{s:04d}.txt")[2] for s in range(S) ]  
            n_ws = read_mesh_info_surface(path_to_mesh+f"/{mesh_type}mesh_info_S0000.txt")[0]
            l_Gs = read_mesh_info_surface(path_to_mesh+f"/{mesh_type}mesh_info_S0000.txt")[1]
#===================== DOING JJ_S

            mesh_jjs = [ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_jjs_S{s:04d}"+mesh_ext,dtype=np.int32).reshape(n_ws,MEs[s],order="F") for s in range(S) ]

            for s in range(1,S):
                mesh_jjs[s]+=cumul_nodes_per_S[s-1]
            mesh_jjs = np.hstack(mesh_jjs)-1

#==================== DOING WW_S

            mesh_wws = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_wws_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(n_ws,l_Gs,order="F") 
                            for s in range(S) ])
            if S > 1:
                ww_assertion = mesh_wws[:,:l_Gs] == mesh_wws[:,l_Gs:2*l_Gs]
                for s in range(2, S):
                    ww_assertion = np.logical_and(ww_assertion, mesh_wws[:,:l_Gs]==mesh_wws[:,s*l_Gs:(s+1)*l_Gs])
                assert np.prod(ww_assertion)
            mesh_wws=mesh_wws[:,:l_Gs]

#==================== DOING RJ_S

            mesh_rjs = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_rjs_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(l_Gs,MEs[s],order="F") 
                                    for s in range(S) ])
#==================== DOING DW_S
            mesh_dws = np.concatenate([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_dws_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(1,n_ws,l_Gs,MEs[s],order="F") 
                            for s in range(S) ], axis=3)
        
            self.jjs = mesh_jjs
            self.wws = mesh_wws
            self.dws = mesh_dws
            self.rjs = mesh_rjs
            self.nws = n_ws
            self.l_Gs = l_Gs
            self.mes = mesh_jjs.shape[1]
            self.surface = True
        else:
            self.surface = False

    def find_duplicate(self, hmin=1e-5):
        print("Looking for duplicates in mesh")
    #============== FINDING DUPLICATE INDICES
        indices = []
        for i in range(self.jj.max()+1):
            r,z = self.R[i], self.Z[i]

            test_tab = np.delete(np.array([self.R, self.Z]), i, axis=1)
            if np.sqrt(np.sum((np.array([r, z]).reshape(2, 1)-test_tab)**2, axis=0)).min()<hmin:
                tab = np.where(np.sqrt(np.sum((np.array([r, z]).reshape(2, 1)-test_tab)**2, axis=0))<hmin)[0]
                if len(tab) != 1:
                    raise ValueError(f"ERROR IN FIND_DUPLICATE in loop {i}: found too many mixed points, check mesh.R[{tab}] and mesh.Z[{tab}]")
                # assert len(tab) == 1
                if tab[0] >= i:
                    j = tab[0] + 1
                    new_pair = np.array([i, j])
                else:
                    j = tab[0]
                    new_pair = np.array([j, i])
                # mask = tab>i
                # tab[mask] += 1
                # if not new_pair in indices:
                indices.append(new_pair)
            # if i%20000 == 0:
            #     print(f"doing {i}")

        indices = np.array(indices)

        self.tab_duplicates = indices[:, :]
        # self.tab_rm = indices[:, 1]

    def rm_duplicate(self,hmin=1e-5):

        if self.tab_duplicates is None:
            self.find_duplicate()

    #============== REMOVING DUPLICATES FROM R & Z
        # self.R = np.delete(self.R, indices[:, 1])
        # self.Z = np.delete(self.Z, indices[:, 1])
        self.R = np.delete(self.R, self.tab_duplicates[:, 1])
        self.Z = np.delete(self.Z, self.tab_duplicates[:, 1])
    #============== GIVING PROPER INDICES FOR mesh.jj
        for i in range(len(indices)):
            j = self.tab_duplicates[i, 1]
            tab_replace = np.where(self.jj==j)
            self.jj[tab_replace] = self.tab_duplicates[i, 0]

            if self.surface:
                tab_replace = np.where(self.jjs==j)
                self.jjs[tab_replace] = self.tab_duplicates[i, 0]


    def build_tab_sym(self, epsilon_z_0=1e-7):

        if self.tab_duplicates is None:
            self.find_duplicate()
            # raise NameError('Make sure to first remove the duplicates')
        
        # partial_sort = np.argsort(self.R**3+self.Z**2)
        # mask_z_is_not_0 = np.abs(self.Z[partial_sort])>epsilon_z_0
        
        # flip_partial_sort = np.copy(partial_sort)
        # restriction_z_is_not_0 = np.zeros(mask_z_is_not_0.sum(), dtype=np.int32)

        # restriction_z_is_not_0[1::2] = partial_sort[mask_z_is_not_0][::2]
        # restriction_z_is_not_0[::2] = partial_sort[mask_z_is_not_0][1::2]

        # flip_partial_sort[mask_z_is_not_0] = restriction_z_is_not_0

        # inverse_partial_sort = np.empty(partial_sort.shape[0],dtype=np.int32)
        # inverse_partial_sort[partial_sort] = np.arange(partial_sort.shape[0])

        # tab_sym = flip_partial_sort[inverse_partial_sort]

        # self.tab_sym = tab_sym
        print("Building tab_sym")
        alt_R, alt_Z = np.delete(self.R, self.tab_duplicates[:, 1]), np.delete(self.Z, self.tab_duplicates[:, 1])
        epsilon_z_0 = 1e-7


        partial_sort = np.argsort(alt_R**3+alt_Z**2)
        mask_z_is_not_0 = np.abs(alt_Z[partial_sort])>epsilon_z_0

        flip_partial_sort = np.copy(partial_sort)
        restriction_z_is_not_0 = np.zeros(mask_z_is_not_0.sum(), dtype=np.int32)

        restriction_z_is_not_0[1::2] = partial_sort[mask_z_is_not_0][::2]
        restriction_z_is_not_0[::2] = partial_sort[mask_z_is_not_0][1::2]

        flip_partial_sort[mask_z_is_not_0] = restriction_z_is_not_0

        inverse_partial_sort = np.empty(partial_sort.shape[0],dtype=np.int32)
        inverse_partial_sort[partial_sort] = np.arange(partial_sort.shape[0])

        alt_tab_sym = flip_partial_sort[inverse_partial_sort]

        tab_sym = np.empty(self.R.shape, dtype=np.int32)
        mask = np.ones(self.R.shape, dtype=bool)
        mask[self.tab_duplicates[:, 1]] = False
        tab_sym[mask] = alt_tab_sym
        tab_sym[self.tab_duplicates[:, 1]] = tab_sym[self.tab_duplicates[:, 0]]
        self.tab_sym = tab_sym

#==================================================================================
#==================================================================================
#=======================  DEFINING SFEMANS PARAMETER OBJECT  ======================
#==================================================================================
#==================================================================================
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
        search_string = f"{mesh_type}mesh_rr_"

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

        search_string = f"{mesh_type}mesh_rr_"
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

    def add_suite_ns(self, path_suites, name_suites="suite_ns_", field=None, D=None, replace = False):
        if self.bins == True:
            raise Exception("your SFEMaNS parameter does already contain information about binaries, please create a new one for suites")
        if self.suites == True:
            if not replace:
                raise ValueError("this SFEMaNS_par already contains data, please set 'replace=True'")
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

    def add_suite_maxwell(self, path_suites, name_suites="suite_maxwell_", field=None, D=None, replace=False):
        if self.bins == True:
            raise Exception("your SFEMaNS parameter does already contain information about binaries, please create a new one for suites")
        if self.suites == True:
            if not replace:
                raise ValueError("this SFEMaNS_par already contains data, please set 'replace=True'")
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

    def add_bins(self, path_binaries, field=None, D=None, replace=False, from_gauss=False):
        if self.suites == True:
            raise Exception("your SFEMaNS parameter does already contain information about suites, please create a new one for binaries")
        if self.bins == True:
            if not replace:
                raise ValueError("this SFEMaNS_par already contains data, please set 'replace=True'")
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
                    #if (f"{test_field}1" in matching_files) and (f"{test_field}2" in matching_files) and (f"{test_field}3" in matching_files):
                    #    list_possible_fields.append((elm[:len_char-1], 3))
                    #else:
                    #    list_possible_fields.append((elm, 1))
                    list_possible_fields.append(elm)

                list_possible_fields = list(set(list_possible_fields)) 

            search_string_1 = "fourier_"
            search_string_2 = "_S0000_F0000"
            #search_string_2 = "c_S0000_F0000"
            fourier_per_mode = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))])>0
            if fourier_per_mode:
                matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
                matching_files = [elm.split(search_string_1)[1].split(search_string_2)[0] for elm in matching_files]
                matching_files = list(set(matching_files))
                for i,elm in enumerate(matching_files):
                    #if elm[-1] == '1' and i+2<len(matching_files) and matching_files[i+1][-1] == '2' and matching_files[i+2][-1] == '3':
                    #    list_possible_fields.append((elm[:-1], 3))
                    #elif elm[-1] == '2' and i+1<len(matching_files) and matching_files[i+1][-1] == '3' and (i-1 >= 0) and matching_files[i-1][-1] == '1':
                    #    continue
                    #elif elm[-1] == '3' and (i-2 >= 0) and matching_files[i-1][-1] == '2' and matching_files[i-2][-1] == '1':
                    #    continue
                    #else:
                    #    list_possible_fields.append((elm, 1))
                    list_possible_fields.append(elm)

                list_possible_fields = list(set(list_possible_fields)) 
                
            search_string_1 = "fourier_"
            search_string_2 = "_S0000_I"
            #search_string_2 = "c_S0000_I"

            fourier = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))])>0
            if fourier:
                matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
                matching_files = [elm.split(search_string_1)[1].split(search_string_2)[0] for elm in matching_files]
                matching_files = list(set(matching_files))
                
                for i,elm in enumerate(matching_files):
                    
                    len_char = len(elm)
                    test_field = elm[:len_char-1]
                    #if (f"{test_field}1" in matching_files) and (f"{test_field}2" in matching_files) and (f"{test_field}3" in matching_files):
                    #    list_possible_fields.append((elm[:len_char-1], 3))
                    #else:
                    #    list_possible_fields.append((elm, 1))
                    list_possible_fields.append(elm)

            #field = input(f"possible fields and their dimension {list_possible_fields}")
            field = input(f"possible fields {list_possible_fields}")
            D = int(input("associated dimension"))
            bool_found_field = False
            i = 0
#            while not bool_found_field:
#                if list_possible_fields[i][0] == field:
#                    bool_found_field = True
#                    D = list_possible_fields[i][1]
#                else:
#                    i += 1

        else:
            if D is None:
                raise TypeError(f"you selected the field {field}, please write its dimension D as well")

        search_string = "phys"
        phys = len([file.name for file in Path(f"{path_binaries}/").iterdir() if (search_string in file.name) and (field in file.name)])>0
        
        search_string_1 = "fourier_"
        search_string_2 = "_S0000_F0000"
        #search_string_2 = "c_S0000_F0000"
        fourier_per_mode = len([file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (field in file.name) and (self.mesh_ext in file.name))])>0
        
        search_string_1 = "fourier_"
        search_string_2 = "_S0000_I"
        #search_string_2 = "c_S0000_I"
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
            
            #N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            #N = len(np.fromfile(self.path_to_mesh+f'/{self.mesh_type}mesh_gauss_rj_S0000{self.mesh_ext}'))
            if from_gauss:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_gauss_rj_S0000"+self.mesh_ext))
            else:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_rr_node_S0000"+self.mesh_ext))
            #if D == 3:
            #    char = '1'
            #elif D == 1:
            #    char = ''
            #directory = f'{path_binaries}/phys_{field}{char}_S0000_I{tab_I[0]:04d}{self.mesh_ext}'
            directory = f'{path_binaries}/phys_{field}_S0000_I{tab_I[0]:04d}{self.mesh_ext}'

            mode = np.fromfile(directory)
            MF = ((len(mode)//N//D)+1)//2            
            #MF = ((len(mode)//N)+1)//2            

        if fourier:
            search_string_1 = f"fourier_{field}"
            search_string_2 = "_S0000_I"
            #search_string_2 = "c_S0000_I"
            matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]
            
            # tab_I = np.sort(np.array([int((elm.split('.',0)).split('I')[-1])]))
            tab_I = [int((elm.split(self.mesh_ext)[0]).split(search_string_2)[-1]) for elm in matching_files]
            tab_I = list(set(tab_I))
            tab_I = np.sort(np.array(tab_I))
            
            if from_gauss:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_gauss_rj_S0000"+self.mesh_ext))
            else:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_rr_node_S0000"+self.mesh_ext))
            #N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}mesh_gauss_rj_S0000{self.mesh_ext}'))
            #N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            #if D == 3:
            #    char = '1'
            #elif D == 1:
            #    char = ''
            #directory = f'{path_binaries}/fourier_{field}{char}c_S0000_I{tab_I[0]:04d}{self.mesh_ext}'
            directory = f'{path_binaries}/fourier_{field}_S0000_I{tab_I[0]:04d}{self.mesh_ext}'
            mode = np.fromfile(directory)
            if (not MF is None) and MF != len(mode)//N//D//2:
                raise ValueError(f"phys file and fourier files do not match (computed MF = {MF} on one hand and MF = {len(mode)//N//D//2} on the other hand)")
            else:
                MF = len(mode)//N//D//2
                #MF = len(mode)//N

        if fourier_per_mode:
            search_string_1 = f"fourier_{field}"
            search_string_2 = "_S0000_F"
            matching_files = [file.name for file in Path(f"{path_binaries}/").iterdir() if ((search_string_1 in file.name) and (search_string_2 in file.name) and (self.mesh_ext in file.name))]

            if (not MF is None) and MF != len(matching_files):
            #if (not MF is None) and MF != len(matching_files)//2//D:
                raise ValueError(f"phys/fourier and fourier_per_mode files do not match (computed MF = {MF} on one hand and MF = {len(matching_files)} on the other hand)")
            else:
                MF == len(matching_files)//2//D

            if from_gauss:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_gauss_rj_S0000"+self.mesh_ext))
            else:
                N = len(np.fromfile(self.path_to_mesh+f"/{self.mesh_type}mesh_rr_node_S0000"+self.mesh_ext))
            #N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}mesh_gauss_rj_S0000{self.mesh_ext}'))
            #N = len(np.fromfile(self.path_to_mesh+f'{self.mesh_type}rr_S0000{self.mesh_ext}'))
            #if D == 3:
            #    char = '1'
            #elif D == 1:
            #    char = ''
            #directory = f'{path_binaries}/fourier_{field}{char}c_S0000_F0000{self.mesh_ext}'
            directory = f'{path_binaries}/fourier_{field}_S0000_F0000{self.mesh_ext}'
            mode = np.fromfile(directory)
            nb_I = len(mode)//N//D//2
            #nb_I = len(mode)//N
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
