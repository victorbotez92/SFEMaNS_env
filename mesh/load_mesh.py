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
    return n_w,l_G,me,n_p


class define_mesh:
    def __init__(self, path_to_mesh, mesh_type, mesh_ext = None, import_nodes = True, import_gauss = True):
        directory = Path(path_to_mesh)
        # search_string = f"{mesh_type}rr_"
        search_string = f"{mesh_type}rr_S0000"
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

        search_string_1 = f"{mesh_type}rr_S"
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
        
        if import_gauss:
            R_G = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}rr_S{s:04d}"+mesh_ext).reshape(l_G,ME[s],order="F") for s in range(S)])
            Z_G = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}zz_S{s:04d}"+mesh_ext).reshape(l_G,ME[s],order="F") for s in range(S)])
            W = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+mesh_ext).reshape(l_G,ME[s],order="F") for s in range(S)])
            self.R_G = R_G
            self.Z_G = Z_G
            self.W = W
        if import_nodes:
            R_node = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}mesh_rr_node_S{s:04d}"+mesh_ext) for s in range(S)])
            Z_node = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}mesh_zz_node_S{s:04d}"+mesh_ext) for s in range(S)])
            self.R_node = R_node
            self.Z_node = Z_node
        # self.mesh_type = mesh_type

        if import_gauss and import_nodes:
            mesh_jj = [ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_jj_S{s:04d}"+mesh_ext,dtype=np.int32).reshape(n_w,ME[s],order="F") 
                   for s in range(S) ]
            nodes_per_S = np.array([mesh_jj[s].max() for s in range(S)])
            cumul_nodes_per_S=np.cumsum(nodes_per_S)
            for s in range(1,S):
                mesh_jj[s]+=cumul_nodes_per_S[s-1]
            mesh_jj = np.hstack(mesh_jj)-1
            mesh_ww = np.hstack([ np.fromfile(path_to_mesh+f"/{mesh_type}mesh_gauss_ww_S{s:04d}"+mesh_ext,dtype=np.float64).reshape(n_w,l_G,order="F") 
                            for s in range(S) ])
            if S > 1:
                ww_assertion = mesh_ww[:,:l_G] == mesh_ww[:,l_G:2*l_G]
                for s in range(2, S):
                    ww_assertion = np.logical_and(ww_assertion, mesh_ww[:,:l_G]==mesh_ww[:,s*l_G:(s+1)*l_G])
                assert np.prod(ww_assertion)
            mesh_ww=mesh_ww[:,:l_G]
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

    def help(self):
        print("path_to_mesh, mesh_ext, S, ME, n_w, l_G")
        print("R_G, Z_G, W")
        print("R_node, Z_node")
        print("jj, ww, dw, rj, nw, lG, me, nn")