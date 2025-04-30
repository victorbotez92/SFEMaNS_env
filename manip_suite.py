import numpy as np
from einops import rearrange,repeat
from scipy.interpolate import griddata

from read_stb import get_mesh

    #=========== Converting from "phys" format to "fourier" format

def contrib_FFT(phys_data, mF, a, MF,shift = 0):
    if mF == 0:
        coeff = 1
    else:
        coeff = 2

    nb_angles = np.shape(phys_data)[1]
    angles = 2*np.pi*np.arange(nb_angles)/nb_angles

    if a == 0:
        trigo = np.cos(mF*(angles+shift))
    elif a == 1:
        trigo = np.sin(mF*(angles+shift))
    trigo = repeat(trigo, 'T -> d T n', d = np.shape(phys_data)[0], n = np.shape(phys_data)[2])

    return coeff*np.mean(trigo*phys_data[:, :, :],axis=1)


def phys_to_fourier(sfem_par,field_in,MF_in=[],shift=0): #requires "phys" format (D theta N), outputs "Fourier" format (D (MF a) N)
    shape_field = np.shape(field_in)
    D = shape_field[0]
    N = shape_field[2]

    if isinstance(MF_in, int):
        MF_in = list(MF_in)
    if len(MF_in) == 0:
        nb_mF = (shape_field[1]+1)//2
        MF_in = np.arange(nb_mF)
        print(f"WARNING: you didn't type MF, setting to arange({nb_mF}) by default")

    field_out = np.zeros((D, 2*len(MF_in), N))
    for num_mF,mF in enumerate(MF_in):
        if num_mF%10 == 0:
            print(f"Doing {num_mF} out of {len(MF_in)}")
        for a in range(2):
            if mF == 0 and a == 1:
                continue
            if mF == 0:
                coeff = 1
            else:
                coeff = 2
            fourier_mode = contrib_FFT(field_in[:, :, :], mF, a, MF_in, shift=shift)
            field_out[:, a+2*num_mF, :] = fourier_mode
    return field_out

    #=========== Converting from "fourier" format to "phys" format

def contrib_IFT(fourier_data, mF, a, MF, shift=0):
    angles = 2*np.pi*np.arange(2*MF-1)/(2*MF-1)
    if a == 0:
        trigo = np.cos(mF*(angles+shift))
    elif a == 1:
        trigo = np.sin(mF*(angles+shift))
    new_contrib = repeat(fourier_data,'d n -> d t n',t = len(trigo))*repeat(trigo,'t -> d t n', d=np.shape(fourier_data)[0], n = np.shape(fourier_data)[1])
    return new_contrib

def fourier_to_phys(sfem_par,field_in,MF_in=[],MF_max=-1,shift=0): #requires "fourier" format (D (MF a) N), outputs "phys" format (D theta N)
    shape_field = np.shape(field_in)
    D = shape_field[0]
    N = shape_field[2]

    if MF_max == -1:
        MF_max = sfem_par.MF
        print(f'WARNING: you did not select MF_max, setting it to {sfem_par.MF} by default')

    if isinstance(MF_in, int):
        MF_in = list(MF_in)

    if len(MF_in) == []:
        nb_mF = shape_field[1]//2
        MF_in = np.arange(nb_mF)
        MF_max = MF_in[-1]
        print(f'WARNING: you did not select MF_max, setting it to {MF_in[-1]} by default')
        print(f'WARNING: you did not select MF_in, setting it to arange({MF_in[-1]}) by default')

    else:
        nb_mF = len(MF_in)
    # try:
    #     assert nb_mF == shape_field[1]//2
    # except AssertionError:
    #     raise IndexError("MF_in must match second dimension of field_in")

    field_out = np.zeros((D, 2*MF_max-1, N))
    for num_mF,mF in enumerate(MF_in):
        if num_mF%10 == 0:
            print(f"Doing {num_mF} out of {len(MF_in)}")
        for a in range(2):
            if mF == 0 and a == 1:
                continue
            new_contrib = contrib_IFT(field_in[:, a+2*num_mF, :], mF, a, MF_max,shift=shift)
            field_out[:, :, :] += new_contrib
    
    return field_out


#=========== saving in "phys" format

def save_phys_from_func(sfem_par,calc_func,path_out,field_name='',iteration_num=0):
    if field_name == '':
        field_name = sfem_par.field
    for s in range(sfem_par.S):
        R = np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext)
        Z = np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}zz_S{s:04d}"+sfem_par.mesh_ext)
        for d in range(sfem_par.D):
            print(f'doing s = {s} and d = {d}')
            component_d_section_s = calc_func(d,R,Z)
            if sfem_par.D == 3:
                file_name = f'phys_{field_name}{d+1}_S{s:04d}_I{iteration_num:04d}{sfem_par.mesh_ext}'
            elif sfem_par.D == 1:
                file_name = f'phys_{field_name}_S{s:04d}_I{iteration_num:04d}{sfem_par.mesh_ext}'
            with open(path_out+file_name,"wb") as f:
                f.write(np.ascontiguousarray(component_d_section_s)) # needs to be (theta,N(s))


def save_phys_from_field(sfem_par,field,path_out,field_name='',iteration_num=0): #requires (D theta N)
    if field_name == '':
        field_name = sfem_par.field
    n = 0
    for s in range(sfem_par.S):
        dn = len(np.fromfile(sfem_par.path_to_mesh+f"/{sfem_par.mesh_type}rr_S{s:04d}"+sfem_par.mesh_ext))
        for d in range(sfem_par.D):
            print(f'doing s = {s} and d = {d}')
            component_d_section_s = field[d,:,n:n+dn]
            if sfem_par.D == 3:
                file_name = f'phys_{field_name}{d+1}_S{s:04d}_I{iteration_num:04d}{sfem_par.mesh_ext}'
            elif sfem_par.D == 1:
                file_name = f'phys_{field_name}_S{s:04d}_I{iteration_num:04d}{sfem_par.mesh_ext}'
            with open(path_out+file_name,"wb") as f:
                f.write(np.ascontiguousarray(component_d_section_s)) # needs to be (theta,N(s))
        n += dn


# def interpol_sfemans_to_semtex(R_sf,Z_sf,R_se,Z_se,data,method='cubic'):
#     new_data = griddata((R_sf,Z_sf), data, (R_se, Z_se), method = method, fill_value=0)
#     return new_data

# def interpol_semtex_to_sfemans(R_sf,Z_sf,R_se,Z_se,data,method='cubic'):
#     # Flatten the grid into coordinate pairs
#     old_points = np.column_stack([(R_se.T).ravel(), (Z_se.T).ravel()])
#     old_values = data.ravel()

#     new_points = np.column_stack([R_sf, Z_sf])
#     new_values = griddata(old_points, old_values, new_points, method=method)
#     return new_values

#========= Interpolating between SFEMaNS (triangle mesh) and rectangular mesh in R and Z

    #============== Interpolation of SFEMaNS mesh on uniform R,Z


def indiv_interpol_sfemans_to_uniform(R_sfem,Z_sfem,new_R,new_Z,field,method='cubic'):
    new_data = griddata((R_sfem,Z_sfem), field, (new_R, new_Z), method = method)#, fill_value=0)
    return new_data


def interpol_sfemans_to_uniform(sfem_par,field,R_linspace,Z_linspace,method='cubic'): #requires either phys (D theta N), fourier (D (MF*a) N) or fourier_per_mode (T D a N)
    R, Z, _ = get_mesh(sfem_par)
    new_R, new_Z = np.meshgrid(R_linspace, Z_linspace)
    
    shape_field = np.shape(field)
    if len(shape_field) == 3:
        azimuth = shape_field[1]
        D = shape_field[0]
        new_field = np.empty((D,azimuth,len(R_linspace),len(Z_linspace)))
    elif len(shape_field) == 4:
        azimuth = shape_field[2]
        D = shape_field[1]
        T = shape_field[0]
        new_field = np.empty((T,D,azimuth,len(R_linspace),len(Z_linspace)))
    else:
        raise TypeError("Unkown format for field")

    for d in range(D):
        for a in range(azimuth):
            if len(shape_field) == 3:
                data_to_interpolate = field[d, a, :]
                print(f'interpolating azimuth {a} out of {azimuth}')
                data_interpolated = indiv_interpol_sfemans_to_uniform(R,Z,new_R,new_Z,data_to_interpolate,method=method)
                new_field[d, a, :, :] = data_interpolated.T

            elif len(shape_field) == 4:
                for t in range(T):
                    data_to_interpolate = field[t, d, a, :]
                    print(f'interpolating azimuth {a} out of {azimuth}, at t = {t} out of {T}')
                    data_interpolated = indiv_interpol_sfemans_to_uniform(R,Z,new_R,new_Z,data_to_interpolate,method=method)
                    new_field[t, d, a, :, :] = data_interpolated.T
            else:
                raise TypeError("Type of field not yet programmed for interpolation")
    return new_field

    #============== Interpolation of uniform R,Z on SFEMaNS mesh

def indiv_interpol_uniform_to_sfemans(R_sfem,Z_sfem,R_linspace,Z_linspace,field,method='cubic'):
    old_points = np.column_stack([(R_linspace.T).ravel(), (Z_linspace.T).ravel()])
    old_values = field.ravel()

    new_points = np.column_stack([R_sfem, Z_sfem])
    new_values = griddata(old_points, old_values, new_points, method=method)
    return new_values

def interpol_uniform_to_sfemans(sfem_par,field,R_linspace,Z_linspace,method='cubic'): #requires either phys (D theta R Z), fourier (D (MF*a) R Z) or fourier_per_mode (T D a R Z)
    R, Z, _ = get_mesh(sfem_par)
    new_R, new_Z = np.meshgrid(R_linspace, Z_linspace)
    
    shape_field = np.shape(field)

    if len(shape_field) == 4:
        z = shape_field[3]
        r = shape_field[2]
        azimuth = shape_field[1]
        D = shape_field[0]
        new_field = np.empty((D,azimuth,len(R)))
    elif len(shape_field) == 5:
        z = shape_field[4]
        r = shape_field[3]
        azimuth = shape_field[2]
        D = shape_field[1]
        T = shape_field[0]
        new_field = np.empty((T,D,azimuth,len(R)))
    else:
        raise TypeError("Unkown format for field")

    try:
        assert r == len(R_linspace) and z == len(Z_linspace)
    except AssertionError:
        raise IndexError("the uniform meshes considered do not match the shape of input field")

    for d in range(D):
        for a in range(azimuth):
            if len(shape_field) == 4:
                data_to_interpolate = field[d, a, :, :]
                if a%10 == 0:
                    print(f'Doing d = {d}: interpolating azimuth {a} out of {azimuth}')
                data_interpolated = indiv_interpol_uniform_to_sfemans(R,Z,new_R,new_Z,data_to_interpolate,method=method)
                new_field[d, a, :] = data_interpolated

            elif len(shape_field) == 5:
                for t in range(T):
                    data_to_interpolate = field[t, d, a, :, :]
                    if a%10 == 0:
                        print(f'interpolating azimuth {a} out of {azimuth}, at t = {t} out of {T}')
                    data_interpolated = indiv_interpol_uniform_to_sfemans(R,Z,new_R,new_Z,data_to_interpolate,method=method)
                    new_field[t, d, a, :] = data_interpolated
            else:
                raise TypeError("Type of field not yet programmed for interpolation")
    return new_field


    #============== Interpolation of SFEMaNS mesh on SFEMaNS mesh



def indiv_interpol_sfemans_to_sfemans(R_sfem,Z_sfem,new_R_sfem,new_Z_sfem,field,method='cubic'):
    new_points = np.column_stack([new_R_sfem, new_Z_sfem])
    new_data = griddata((R_sfem,Z_sfem), field, new_points, method = method)#, fill_value=0)
    return new_data


def interpol_sfemans_to_sfemans(sfem_par,field,new_R_sfem,new_Z_sfem,method='cubic'): #requires either phys (D theta N), fourier (D (MF*a) N) or fourier_per_mode (T D a N)
    R, Z, _ = get_mesh(sfem_par)
    # new_R, new_Z = np.meshgrid(R_linspace, Z_linspace)
    
    shape_field = np.shape(field)
    if len(shape_field) == 3:
        azimuth = shape_field[1]
        D = shape_field[0]
        new_field = np.empty((D,azimuth,len(R_linspace),len(Z_linspace)))
    elif len(shape_field) == 4:
        azimuth = shape_field[2]
        D = shape_field[1]
        T = shape_field[0]
        new_field = np.empty((T,D,azimuth,len(new_R_sfem)))
    else:
        raise TypeError("Unkown format for field")

    for d in range(D):
        for a in range(azimuth):
            if len(shape_field) == 3:
                data_to_interpolate = field[d, a, :]
                if a%10 == 0:
                    print(f'interpolating azimuth {a} out of {azimuth}')
                data_interpolated = indiv_interpol_sfemans_to_uniform(R,Z,new_R_sfem,new_Z_sfem,data_to_interpolate,method=method)
                new_field[d, a, :] = data_interpolated

            elif len(shape_field) == 4:
                for t in range(T):
                    data_to_interpolate = field[t, d, a, :]
                    if a%10 == 0:
                        print(f'interpolating azimuth {a} out of {azimuth}, at t = {t} out of {T}')
                    data_interpolated = indiv_interpol_sfemans_to_uniform(R,Z,new_R_sfem,new_Z_sfem,data_to_interpolate,method=method)
                    new_field[t, d, a, :] = data_interpolated
            else:
                raise TypeError("Type of field not yet programmed for interpolation")
    return new_field