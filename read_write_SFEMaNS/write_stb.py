import numpy as np

#=========== saving in "fourier per mode format"

def write_fourier_per_mode(sfem_par,field,path_out,field_name='',mF=0): #field is shape (N a*D T)
    """
    Function that writes a field of shape (N a*D T) in SFEMaNS Fourier_per_mode format
    args:
        sfem_par: SFEMaNS parameter (contains mesh_type, mesh_ext, path_to_mesh information)
        field: the one to be saved, of shape (N a*D T)
        path_out
        field_name
        mF: Fourier mode (the two axis cos & sine are always saved together)
    """

    if field.shape[1] == 3 or field.shape[1] == 1:
        raise TypeError("Error in write_fourier_per_mode: it seems like your field is in phys format and not fourier")
    
    if field_name == '':
        field_name = sfem_par.field
    n = 0

    D = field.shape[1]//2

    for s in range(sfem_par.S):
        dn = len(np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext))
        for d in range(D):
            for a,axis in enumerate(['c','s']):
                if D > 1:
                    file_name = "/fourier_{f}{i}{ax}_S{s:04d}_F{m:04d}".format(f=field_name,i=d+1,ax=axis,s=s,m=mF)+sfem_par.mesh_ext
                elif D == 1:
                    file_name = "/fourier_{f}{ax}_S{s:04d}_F{m:04d}".format(f=field_name,ax=axis,s=s,m=mF)+sfem_par.mesh_ext
                component_d_section_s_fourier_mF_a = field[n:n+dn, a+2*d, :]
                with open(path_out+file_name,"wb") as f:
                    f.write(np.ascontiguousarray(component_d_section_s_fourier_mF_a.T)) # needs to be (T, N(s))
        n += dn

#=========== saving in "fourier format"

def write_fourier(sfem_par,field,path_out,field_name='',I=0): #field is shape (N a*D MF)
    """
    Function that writes a field of shape (N a*D MF) in SFEMaNS Fourier format
    args:
        sfem_par: SFEMaNS parameter (contains mesh_type, mesh_ext, path_to_mesh information)
        field: the one to be saved, of shape (N a*D MF)
        path_out
        field_name
        I: value of iteration to be saved
    """
    if field.shape[1] == 3 or field.shape[1] == 1:
        raise TypeError("Error in write_fourier: it seems like your field is in phys format and not fourier")
    if field_name == '':
        field_name = sfem_par.field
    n = 0

    D = field.shape[1]//2

    for s in range(sfem_par.S):
        dn = len(np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext))
        for d in range(D):
            for a,axis in enumerate(['c','s']):
                if D > 1:
                    file_name = "/fourier_{f}{i}{ax}_S{s:04d}_I{m:04d}".format(f=field_name,i=d+1,ax=axis,s=s,m=I)+sfem_par.mesh_ext
                elif D == 1:
                    file_name = "/fourier_{f}{ax}_S{s:04d}_I{m:04d}".format(f=field_name,ax=axis,s=s,m=I)+sfem_par.mesh_ext
                component_d_section_s_fourier_mF_a = field[n:n+dn, a+2*d, :]
                with open(path_out+file_name,"wb") as f:
                    f.write(np.ascontiguousarray(component_d_section_s_fourier_mF_a.T)) # needs to be (MF, N(s))
        n += dn

#=========== saving in "phys" format

def save_phys_from_func(sfem_par,calc_func,path_out,field_name='',I=0):
    if field_name == '':
        field_name = sfem_par.field
    for s in range(sfem_par.S):
        R = np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext)
        Z = np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}zz_S{s:04d}"+sfem_par.mesh_ext)
        for d in range(sfem_par.D):
            print(f'doing s = {s} and d = {d}')
            component_d_section_s = calc_func(d,R,Z)
            if sfem_par.D == 3:
                file_name = f'phys_{field_name}{d+1}_S{s:04d}_I{I:04d}{sfem_par.mesh_ext}'
            elif sfem_par.D == 1:
                file_name = f'phys_{field_name}_S{s:04d}_I{I:04d}{sfem_par.mesh_ext}'
            with open(path_out+file_name,"wb") as f:
                f.write(np.ascontiguousarray(component_d_section_s.T)) # needs to be (theta,N(s))


def write_phys(sfem_par,field,path_out,field_name='',I=0): #requires (N D theta)
    """
    Function that writes a field of shape (N D theta) in SFEMaNS Fourier format
    args:
        sfem_par: SFEMaNS parameter (contains mesh_type, mesh_ext, path_to_mesh information)
        field: the one to be saved, of shape (N D theta) !!! if MF Fourier modes, then there must be 2*MF-1 angles !!!
        path_out
        field_name
        I: value of iteration to be saved
    """
    if field.shape[1] == 6 or field.shape[1] == 2:
        raise TypeError("Error in write_phys: it seems like your field is in fourier format and not phys")
    if field_name == '':
        field_name = sfem_par.field
    n = 0
    for s in range(sfem_par.S):
        dn = len(np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext))
        for d in range(sfem_par.D):
            print(f'doing s = {s} and d = {d}')
            component_d_section_s = field[n:n+dn,d,:]
            if sfem_par.D == 3:
                file_name = f'phys_{field_name}{d+1}_S{s:04d}_I{I:04d}{sfem_par.mesh_ext}'
            elif sfem_par.D == 1:
                file_name = f'phys_{field_name}_S{s:04d}_I{I:04d}{sfem_par.mesh_ext}'
            with open(path_out+file_name,"wb") as f:
                f.write(np.ascontiguousarray(component_d_section_s.T)) # needs to be (theta,N(s))
        n += dn