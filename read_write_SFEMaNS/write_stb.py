import numpy as np



#=========== saving in "fourier format"

def write_fourier(sfem_par,field,path_out,field_name='',I=0): #field is shape (N a*D MF)
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