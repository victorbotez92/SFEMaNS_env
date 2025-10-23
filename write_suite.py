import numpy as np


def write_suite_ns(sfem_par, vv_mesh, pp_mesh,  path_out, field_name="suite_ns", I=0,
                   un = None, un_m1 = None, pn = None, pn_m1 = None, incpn = None, incpn_m1 = None):
    """
    function to write a vector field in SFEMaNS suite_ns format
    Necessarily takes as argument the field itself on nodes and in Fourier space

        eg : for u, first_offset=2, record_stack_lenght=7
        eg : for p, first_offset=4, record_stack_lenght=7

    Any unspecified field will be by default set as zero
    
    """

    dummy_pp = np.zeros((pp_mesh.nn, 2, sfem_par.MF), dtype=np.float64)
    dummy_vv = np.zeros((vv_mesh.nn, 6, sfem_par.MF), dtype=np.float64)
    
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
    
    n_inf_vv = 0
    n_inf_pp = 0
    for s in range(sfem_par.S):
        n_sup_vv = n_inf_vv + vv_mesh.nn_per_S[s]
        n_sup_pp = n_inf_pp + pp_mesh.nn_per_S[s]

        path = path_out + f"/{field_name}_S{s:03d}_I{I:03d}{sfem_par.mesh_ext}"

        dummy_pp = np.zeros((n_sup_pp - n_inf_pp + 1, 2), dtype=np.float64)
        dummy_vv = np.zeros((n_sup_vv - n_inf_vv + 1, 2), dtype=np.float64)
        
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
            
        with open(path,'wb') as file:

            record_length = 8 + 4 + 4 + 4  # float64 (8 bytes) + 3×int32 (12 bytes)
            file.write(np.int32(record_length).tobytes())
            file.write(np.float64(0).tobytes())
            file.write(np.int32(sfem_par.S).tobytes())
            file.write(np.int32(sfem_par.MF).tobytes())
            file.write(np.int32(1).tobytes())
            file.write(np.int32(record_length).tobytes())
            
            for mF in range(sfem_par.MF):
                print("Writing mF=", mF)
    
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
