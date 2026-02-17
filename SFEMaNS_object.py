import numpy as np
from matplotlib.tri import Triangulation
from einops import rearrange

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
        self.jj = mesh_jj

#============ SORTING triangles for plt.tripcolor
        if self.jj.shape[0] == 6:
            tri_1 = self.jj.T[:, 3:]
            tri_2 = self.jj.T[:, np.array([0, 4, 5])]
            tri_3 = self.jj.T[:, 1::2]
            tri_4 = self.jj.T[:, 2:5]
            
            list_triangles = np.concatenate((tri_1, tri_2, tri_3, tri_4), axis=0)
        elif self.jj.shape[0] == 3:
            list_triangles = np.copy(self.jj.T)

        else:
            raise TypeError(f'Error in define_mesh triangulation => found unknown Pk decomposition with nb points {self.jj.shape[0]}')

        tri_R = self.R[list_triangles]
        tri_Z = self.Z[list_triangles]
        
        vecs_1 = np.zeros((tri_R.shape[0], 2))
        vecs_1[:, 1] = tri_Z[:, 1] - tri_Z[:, 0]
        vecs_1[:, 0] = tri_R[:, 1] - tri_R[:, 0]
        vecs_2 = np.zeros((tri_R.shape[0], 2))
        vecs_2[:, 1] = tri_Z[:, 2] - tri_Z[:, 0]
        vecs_2[:, 0] = tri_R[:, 2] - tri_R[:, 0]
        vec_prod = np.sign(vecs_2[:, 1]*vecs_1[:, 0] - vecs_2[:, 0]*vecs_1[:, 1])
        
        perm_1 = np.copy(list_triangles[vec_prod==-1, :][:, 1])
        perm_2 = np.copy(list_triangles[vec_prod==-1, :][:, 2])
        list_triangles[vec_prod==-1, 1] = perm_2
        list_triangles[vec_prod==-1, 2] = perm_1
        
        tri = Triangulation(self.R, self.Z, triangles = list_triangles)
        self.tri = tri 
        
        tri_R = self.R[list_triangles]
        tri_Z = self.Z[list_triangles]
        
        vecs_1 = np.zeros((tri_R.shape[0], 2))
        vecs_1[:, 1] = tri_Z[:, 1] - tri_Z[:, 0]
        vecs_1[:, 0] = tri_R[:, 1] - tri_R[:, 0]
        vecs_2 = np.zeros((tri_R.shape[0], 2))
        vecs_2[:, 1] = tri_Z[:, 2] - tri_Z[:, 0]
        vecs_2[:, 0] = tri_R[:, 2] - tri_R[:, 0]
        
        vec_prod = np.sign(vecs_2[:, 1]*vecs_1[:, 0] - vecs_2[:, 0]*vecs_1[:, 1])
        
        assert (vec_prod==-1).sum()==0, (vec_prod==-1).sum()
#============== DOING ww
        mesh_ww = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_ww_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(n_w,l_G,order="F") 
                        for s in range(S) ])
        if S > 1:
            ww_assertion = mesh_ww[:,:l_G] == mesh_ww[:,l_G:2*l_G]
            for s in range(2, S):
                ww_assertion = np.logical_and(ww_assertion, mesh_ww[:,:l_G]==mesh_ww[:,s*l_G:(s+1)*l_G])
            assert np.prod(ww_assertion)
        mesh_ww=mesh_ww[:,:l_G]
        self.ww = mesh_ww
#=============== DOING rj
        mesh_rj = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_rj_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(l_G,ME[s],order="F") 
                        for s in range(S) ])
        
        self.rj = mesh_rj

        nw, lG, me = mesh_ww.shape[0], mesh_ww.shape[1], mesh_jj.shape[1]
        self.nw, self.lG, self.me = nw, lG, me

        nn = np.max(mesh_jj) + 1
        self.nn = nn
        self.nn_per_S = nodes_per_S

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

    def find_duplicate(self, hmin=1e-7):
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
        for i in range(len(self.tab_duplicates)):
        #for i in range(len(indices)):
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

def generate_pp_from_vv(vv_mesh):

    vv_jj_cut = vv_mesh.jj[:3, :]
    vv_jj_cut = rearrange(vv_jj_cut, 'nw me -> (nw me)')
    sort_vv = np.argsort(vv_jj_cut)

    new_pp_mesh = define_mesh(vv_mesh.path_to_mesh, 'vv')

    sorted_vv_jj = vv_jj_cut[sort_vv]
    new_pp_nn_per_S = np.empty(vv_mesh.nn_per_S.shape, dtype=np.int32)
    diff_sorted_vv_jj = np.concatenate((np.array([0]), np.diff(sorted_vv_jj)))

    offset = 0
    for i, elm in enumerate(vv_mesh.nn_per_S):

        mask_vv_in_S = np.logical_and(diff_sorted_vv_jj > 0, np.logical_and(offset <= sorted_vv_jj, sorted_vv_jj < elm+offset))
        new_pp_nn_per_S[i] = mask_vv_in_S.sum() + (i==0)
        offset += elm


    mask_step = diff_sorted_vv_jj!=0
    diff_sorted_vv_jj[mask_step] -= 1
    diff_sorted_vv_jj = np.cumsum(diff_sorted_vv_jj)
    offset_sorted_vv_jj = sorted_vv_jj-diff_sorted_vv_jj

    new_pp_jj = np.zeros(vv_jj_cut.shape, dtype=np.int32)
    new_pp_jj[sort_vv] = offset_sorted_vv_jj
    new_pp_jj = rearrange(new_pp_jj, '(nw me) -> nw me', me = vv_mesh.me)

    new_pp_R = np.empty(new_pp_jj.max() + 1)
    new_pp_R[new_pp_jj] = vv_mesh.R[vv_mesh.jj[:3, :]]
    new_pp_Z = np.empty(new_pp_jj.max() + 1)
    new_pp_Z[new_pp_jj] = vv_mesh.Z[vv_mesh.jj[:3, :]]

    new_pp_mesh.R = new_pp_R
    new_pp_mesh.Z = new_pp_Z
    new_pp_mesh.jj = new_pp_jj
    new_pp_mesh.l_G = 3
    new_pp_mesh.nw = 3

    new_pp_mesh.ww = None
    new_pp_mesh.rj = None
    new_pp_mesh.dw = None
    new_pp_mesh.nn = np.max(new_pp_mesh.jj) + 1
    new_pp_mesh.nn_per_S = new_pp_nn_per_S

    return new_pp_mesh


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
    """
    class SFEMaNS parameter which is necessary for every operation.
__init__ : takes path_to_mesh and mesh_type (initialize required information for SFEMaNS parameter
add_suite_ns : takes path_suites, name_suites, field, D, replace=False
add_suite_maxwell : same
add_bins : takes path_binaries, field, D, replace=False

One SFEMaNS parameter can only account for one mesh_type and then one path containing suites_ns, or one path with suite_mxw, or one path with any kind of binaries
    """
 
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
        self.MF = None
    def add_suite_ns(self, path_suites, name_suites="suite_ns_", field=None, D=None, replace = False):
        type_vv = ['u', 'un', 'un_m1']
        type_pp = ['p', 'pn', 'pn_m1', 'incp', 'incpn', 'incpn_m1']
        if self.bins == True:
            raise Exception("your SFEMaNS parameter does already contain information about binaries, please create a new one for suites")
        if self.suites == True:
            if not replace:
                raise ValueError("this SFEMaNS_par already contains data, please set 'replace=True'")
        if not self.mesh_type in ["pp", "vv"]:
            raise TypeError(f"SFEMaNS object is on {self.mesh_type} while ns-like suite requires pp or vv")
        if field is None:
            if self.mesh_type == "vv":
                field = input(f"on {self.mesh_type}, possibilities are {type_vv}")
                if not field in type_vv:
                    raise NameError(f"Choose your field in {type_vv}")
                d = 3
            elif self.mesh_type == "pp":
                field = input(f"on {self.mesh_type}, possibilities are {type_pp}")
                if not field in type_pp:
                    raise NameError(f"Choose your field in {type_pp}")
                d = 1

            if (not D is None) and d != D:
                raise ValueError(f"input D={D} differs from D={d} from {field}")
            
        search_string_1 = f"{name_suites}S000_I"
        matching_files = [file.name for file in Path(f"{path_suites}/").iterdir() if ((search_string_1 in file.name) and (self.mesh_ext in file.name))]
        matching_files = [elm.split(search_string_1)[1].split(self.mesh_ext)[0] for elm in matching_files]
        matching_files = list(set(matching_files))
        search_string = f"{name_suites}S000{self.mesh_ext}"
        no_I_matching_file = [file.name for file in Path(f"{path_suites}/").iterdir() if (search_string in file.name)]
        if len(no_I_matching_file) == 1:
            matching_files.append(-1)
        tab_I = np.sort(np.array(matching_files))
        if len(tab_I) == 0:
            raise FileNotFoundError(f"no file of the form {name_suites} found in {path_suites}/ including {self.mesh_ext}")
        
        self.suites = True
        self.type_suite = "ns"
        self.path_suites = path_suites
        self.name_suites = name_suites
        self.field = field
        self.tab_I = tab_I

        list_MF_found = []
        for elm_I in tab_I:
            list_MF_per_I = []
            for s in range(self.S):
                if int(elm_I) != -1:
                    path_I_s = self.path_suites + '/' +  self.name_suites + f"S{s:03d}_" + f"I{elm_I}" + self.mesh_ext
                elif int(elm_I) == -1:
                    path_I_s = self.path_suites + '/' +  self.name_suites + f"S{s:03d}" + self.mesh_ext

                with open(path_I_s,'rb') as file:
                    
                    # get record lenght
                    record_length_bytes = file.read(4)
                    record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
                    
                    #assuming double precision
                    num_elements = record_length // 4
                    field = np.fromfile(file,dtype=np.int32,count=num_elements)
                    time = np.frombuffer(field[:2].tobytes(), dtype=np.float64)[0]
                    nb_procs_S = field[2]
                    nb_procs_F = field[3]
                    size_list_modes = field[4]
                    MF = nb_procs_F * size_list_modes
                    list_MF_per_I.append(MF)
            if np.asarray(list_MF_per_I).std() != 0:
                raise ValueError(f"found the inconsistent following amount of Fourier modes at {elm_I} for the different subsections: {list_MF_per_I}")

            list_MF_found.append(list_MF_per_I[0])  
        list_MF_found = np.asarray(list_MF_found)
        if np.std(list_MF_found) != 0:
            print("WARNING: different amounts of Fourier modes were found.")
            print(f"Lowest is {list_MF_found.min()} at I{tab_I[np.argmin(list_MF_found)]}")
            print(f"Highest is {list_MF_found.max()} at I{tab_I[np.argmax(list_MF_found)]}")

            self.MF = list_MF_found.min()
            self.consistent_MF = False
        else:
            self.MF = list_MF_found[0]
            self.consistent_MF = True




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
        search_string = f"{name_suites}S000{self.mesh_ext}"
        no_I_matching_file = [file.name for file in Path(f"{path_suites}/").iterdir() if (search_string in file.name)]
        if len(no_I_matching_file) == 1:
            matching_files.append(-1)
        tab_I = np.sort(np.array(matching_files))
        
        self.suites = True
        self.type_suite = "maxwell"
        self.path_suites = path_suites
        self.name_suites = name_suites
        self.field = field
        self.tab_I = tab_I
        if len(tab_I) == 0:
            raise FileNotFoundError(f"no file of the form {name_suites} found in {path_suites}/ including {self.mesh_ext}")

        list_MF_found = []
        for elm_I in tab_I:
            list_MF_per_I = []
            for s in range(self.S):
                if int(elm_I) != -1:
                    path_I_s = self.path_suites + '/' +  self.name_suites + f"S{s:03d}_" + f"I{elm_I}" + self.mesh_ext
                elif int(elm_I) == -1:
                    path_I_s = self.path_suites + '/' +  self.name_suites + f"S{s:03d}" + self.mesh_ext

                with open(path_I_s,'rb') as file:
                    
                    # get record lenght
                    record_length_bytes = file.read(4)
                    record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
                    
                    #assuming double precision
                    num_elements = record_length // 4
                    field = np.fromfile(file,dtype=np.int32,count=num_elements)
                    time = np.frombuffer(field[:2].tobytes(), dtype=np.float64)[0]
                    nb_procs_S = field[2]
                    nb_procs_F = field[3]
                    size_list_modes = field[4]
                    MF = nb_procs_F * size_list_modes
                    list_MF_per_I.append(MF)
            if np.asarray(list_MF_per_I).std() != 0:
                raise ValueError(f"found the inconsistent following amount of Fourier modes at {elm_I} for the different subsections: {list_MF_per_I}")

            list_MF_found.append(list_MF_per_I[0])  
        list_MF_found = np.asarray(list_MF_found)
        if np.std(list_MF_found) != 0:
            print("WARNING: different amounts of Fourier modes were found.")
            print(f"Lowest is {list_MF_found.min()} at I{tab_I[np.argmin(list_MF_found)]}")
            print(f"Highest is {list_MF_found.max()} at I{tab_I[np.argmax(list_MF_found)]}")

            self.MF = list_MF_found.min()
            self.consistent_MF = False
        else:
            self.MF = list_MF_found[0]
            self.consistent_MF = True


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

class m_family:
    """
    class designed to define a field (vector or scalar) wrt its family to save memory + computational time
    __init__: family, nb_shifts, MF
    add_data: field
    """
    def __init__(self, family, nb_shifts, MF):
        self.family=family
        self.nb_shifts=nb_shifts
        self.MF = MF
        if family*2%nb_shifts==0:
            self.correlate_cos_sine = False
            list_modes = family+nb_shifts*np.arange(MF)
            self.list_modes_plus = list_modes

        else:
            self.correlate_cos_sine = True
            list_modes_plus = family + nb_shifts*np.arange(MF)
            list_modes_minus = -family + nb_shifts*(1+np.arange(MF))
            list_modes = np.sort(np.concatenate((list_modes_plus, list_modes_minus)))
            list_modes = list_modes[list_modes<MF]
            list_modes_plus = list_modes[(list_modes-family)%nb_shifts==0]
            list_modes_minus = list_modes[(list_modes+family)%nb_shifts==0]
            self.list_modes_plus = list_modes_plus
            self.list_modes_minus = list_modes_minus

        self.list_modes = list_modes

    def init_data(self, mesh, D, on_gauss=True):
        if on_gauss:
            self.data = np.zeros([mesh.l_G*mesh.me, 2*D, self.list_modes])
        else:
            self.data = np.zeros([mesh.R.shape[0], 2*D, self.list_modes])

    def add_data(self, field):
        if self.MF != field.shape[-1]:
            print(f"WARNING: MF in m_family object {self.MF} does not match MF of field {field.shape[-1]}")
        
        self.data = field[:, :, self.list_modes]
