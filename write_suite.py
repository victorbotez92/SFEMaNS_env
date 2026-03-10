import numpy as np
from .SFEMaNS_object import generate_pp_from_vv

def write_suite_ns(sfem_par, path_out, vv_mesh=None, pp_mesh=None, field_name="suite_ns_", I=0,
                   un = None, un_m1 = None, pn = None, pn_m1 = None, incpn = None, incpn_m1 = None,
                   opt_time = None):
    """
    function to write a vector field in SFEMaNS suite_ns format
    Necessarily takes as argument the field itself on nodes and in Fourier space

        eg : for u, first_offset=2, record_stack_lenght=7
        eg : for p, first_offset=4, record_stack_lenght=7

    Any unspecified field will be by default set as zero
    
    """

    if vv_mesh is None:
        raise ValueError('in write_suite_ns: please at least have vv_mesh as parameter')

    if pp_mesh is None:
        pp_mesh = generate_pp_from_vv(vv_mesh)

    list_MF = []

    if not (un is None):
        list_MF.append(un.shape[-1])
    if not (un_m1 is None):
        list_MF.append(un_m1.shape[-1])
    if not (pn is None):
        list_MF.append(pn.shape[-1])
    if not (pn_m1 is None):
        list_MF.append(pn_m1.shape[-1])
    if not (incpn is None):
        list_MF.append(incpn.shape[-1])
    if not (incpn_m1 is None):
        list_MF.append(incpn_m1.shape[-1])
    list_MF = np.asarray(list_MF)
    if len(list_MF) == 0:
        raise IndexError("in write_suite: no suite specified for writing")
    elif list_MF.std() != 0:
        print(f"WARNING in write_suite: some of the fields have different number of Fourier modes, restraining to mF = {list_MF.min()}")
    mF_max = list_MF.min()
    if sfem_par.MF is None:
        sfem_par.MF = mF_max

    dummy_pp = np.zeros((pp_mesh.nn, 2, mF_max), dtype=np.float64)
    dummy_vv = np.zeros((vv_mesh.nn, 6, mF_max), dtype=np.float64)
    
    if un is None:
        un = dummy_vv
    if un_m1 is None:
        un_m1 = dummy_vv
    if pn is None:
        pn = dummy_pp
    if pn_m1 is None:
        pn_m1 = dummy_pp
    if incpn is None:
        incpn = dummy_pp
    if incpn_m1 is None:
        incpn_m1 = dummy_pp
    if opt_time is None:
        time = 0
    else:
        time = opt_time

    n_inf_vv = 0
    n_inf_pp = 0

    for s in range(sfem_par.S):
        n_sup_vv = n_inf_vv + vv_mesh.nn_per_S[s]
        n_sup_pp = n_inf_pp + pp_mesh.nn_per_S[s]
        if I >= 0:
            path = path_out + f"/{field_name}S{s:03d}_I{I:03d}{sfem_par.mesh_ext}"
            print(f"Writing file {field_name}S{s:03d}_I{I:03d}{sfem_par.mesh_ext}")
        elif I == -1:
            path = path_out + f"/{field_name}S{s:03d}{sfem_par.mesh_ext}"
            print(f"Writing file {field_name}S{s:03d}{sfem_par.mesh_ext}")
        else:
            raise IndexError(f"Incorrect input iteration {I} ==> should be positive integer or -1")

        with open(path,'wb') as file:

            record_length = 8 + 4 + 4 + 4  # float64 (8 bytes) + 3×int32 (12 bytes)
            file.write(np.int32(record_length).tobytes())
            file.write(np.float64(opt_time).tobytes())
            #file.write(np.float64(0).tobytes())
            file.write(np.int32(sfem_par.S).tobytes())
            file.write(np.int32(mF_max).tobytes())
            #file.write(np.int32(sfem_par.MF).tobytes())
            file.write(np.int32(1).tobytes())
            file.write(np.int32(record_length).tobytes())
            
            for mF in range(mF_max):
            #for mF in range(sfem_par.MF):
                #print("Writing mF=", mF)
    
                record_length = 4  # int32 (4 bytes)
                file.write(np.int32(record_length).tobytes())
                file.write(np.int32(mF).tobytes())
                file.write(np.int32(record_length).tobytes())
                
#=================== UN =================#
                data = np.asarray(un[n_inf_vv:n_sup_vv, :, mF], dtype=np.float64) #un 
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

#=================== UN_M1 =================#
                data = np.asarray(un_m1[n_inf_vv:n_sup_vv, :, mF], dtype=np.float64) #un_m1
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

#=================== PN =================#
                data = np.asarray(pn[n_inf_pp:n_sup_pp, :, mF], dtype=np.float64) #pn
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

#=================== PN_M1 =================#
                data = np.asarray(pn_m1[n_inf_pp:n_sup_pp, :, mF], dtype=np.float64) #pn_m1
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

#=================== INCPN =================#
                data = np.asarray(incpn[n_inf_pp:n_sup_pp, :, mF], dtype=np.float64) #incpn
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

#=================== ICNPN_M1 =================#
                data = np.asarray(incpn_m1[n_inf_pp:n_sup_pp, :, mF], dtype=np.float64) #incpn_m1
                record_length = data.nbytes
                file.write(np.int32(record_length).tobytes())
                file.write(data.tobytes(order='F'))
                file.write(np.int32(record_length).tobytes())

        n_inf_vv = n_sup_vv
        n_inf_pp = n_sup_pp



def write_suite_maxwell(sfem_par, path_out, H_mesh=None, phi_mesh=None, field_name="suite_maxwell_", I=0,
                   Hn = None, Hn_m1 = None, Bn = None, Bn_m1 = None, phin = None, phin_m1 = None,
                   opt_time = None):
    """
    function to write a vector field in SFEMaNS suite_maxwell format
    Necessarily takes as argument the field itself on nodes and in Fourier space

    Any unspecified field will be by default set as zero
    
    """

    if H_mesh is None:
        raise ValueError('BUG in write_suite_maxwell: please at least have vv_mesh as parameter')

    if phi_mesh is None and not (phin is None and phin_m1 is None):
        raise ValueError('BUG in write_suite_maxwell: phi_mesh must be specified if either phin or phin_m1 need to be written')
    elif not phi_mesh is None:
        if_phi = True
        if (phin is None) and (phin_m1 is None):
            print('WARNING: phi_mesh specified without any phin or phin_m1 => will be set to zero')
    
    list_char_H = ['Hn', 'Hn_m1', 'Bn', 'Bn_m1']
    list_char_phi = ['phin', 'phin_m1']

    dict_fields = {}
    list_fields = [Hn, Hn_m1, Bn, Bn_m1, phin, phin_m1]

    for field, char in zip(list_fields, list_char_H+list_char_phi):
        dict_fields[char] = field

    list_MF = []
    for char_H in list_char_H:
        if not (dict_fields[char_H] is None):
            list_MF.append(dict_fields[char_H].shape[-1])
    if if_phi:
        for char_phi in list_char_phi:
            if not (dict_fields[char_phi] is None):
                list_MF.append(dict_fields[char_phi].shape[-1])
        
    list_MF = np.asarray(list_MF)
    if len(list_MF) == 0:
        raise IndexError("in write_suite_maxwell: no suite specified for writing")
    elif list_MF.std() != 0:
        print(f"WARNING in write_suite: some of the fields have different number of Fourier modes, restraining to mF = {list_MF.min()}")
    mF_max = list_MF.min()
    if sfem_par.MF is None:
        sfem_par.MF = mF_max

    if if_phi:
        dummy_phi = np.zeros((phi_mesh.nn, 2, mF_max), dtype=np.float64)
    dummy_H = np.zeros((H_mesh.nn, 6, mF_max), dtype=np.float64)
    
    for char_H in list_char_H:
        if dict_fields[char_H] is None:
            dict_fields[char_H] = dummy_H
    if if_phi:
        for char_phi in list_char_phi:
            if dict_fields[char_phi] is None:
                dict_fields[char_phi] = dummy_phi

    if opt_time is None:
        time = 0
    else:
        time = opt_time

    n_inf_H = 0
    n_inf_phi = 0

    for s in range(sfem_par.S):
        n_sup_H = n_inf_H + H_mesh.nn_per_S[s]
        if if_phi:
            n_sup_phi = n_inf_phi + phi_mesh.nn_per_S[s]
        if I >= 0:
            path = path_out + f"/{field_name}S{s:03d}_I{I:03d}{sfem_par.mesh_ext}"
            print(f"Writing file {field_name}S{s:03d}_I{I:03d}{sfem_par.mesh_ext}")
        elif I == -1:
            path = path_out + f"/{field_name}S{s:03d}{sfem_par.mesh_ext}"
            print(f"Writing file {field_name}S{s:03d}{sfem_par.mesh_ext}")
        else:
            raise IndexError(f"Incorrect input iteration {I} ==> should be positive integer or -1")

        with open(path,'wb') as file:

            record_length = 8 + 4 + 4 + 4  # float64 (8 bytes) + 3×int32 (12 bytes)
            file.write(np.int32(record_length).tobytes())
            file.write(np.float64(opt_time).tobytes())
            file.write(np.int32(sfem_par.S).tobytes())
            file.write(np.int32(mF_max).tobytes())
            file.write(np.int32(1).tobytes())
            file.write(np.int32(record_length).tobytes())
            
            for mF in range(mF_max):
    
                record_length = 4  # int32 (4 bytes)
                file.write(np.int32(record_length).tobytes())
                file.write(np.int32(mF).tobytes())
                file.write(np.int32(record_length).tobytes())
   


#=================== fields on Hmesh =================#
                for char_H in list_char_H:
                    data = np.asarray(dict_fields[char_H][n_inf_H:n_sup_H, :, mF], dtype=np.float64) 
                    record_length = data.nbytes
                    file.write(np.int32(record_length).tobytes())
                    file.write(data.tobytes(order='F'))
                    file.write(np.int32(record_length).tobytes())

#=================== fields on phimesh =================#
                if if_phi:
                    for char_phi in list_char_phi:
                        data = np.asarray(dict_fields[char_phi][n_inf_phi:n_sup_phi, :, mF], dtype=np.float64) 
                        record_length = data.nbytes
                        file.write(np.int32(record_length).tobytes())
                        file.write(data.tobytes(order='F'))
                        file.write(np.int32(record_length).tobytes())
        
        n_inf_H = n_sup_H
        n_inf_phi = n_sup_phi
