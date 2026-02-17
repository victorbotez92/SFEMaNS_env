from os import listdir
from os.path import isfile, join
import numpy as np
import sys

from einops import rearrange, einsum

def skip_record(file,n):
    """
    file : file pointer (with open(path,'rb') as file)
    n : number of record to skip
    """
    for _ in range(n):
         # skip first line : read record lenght, go to the next record
        record_length_bytes = file.read(4)
        record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
        file.seek(record_length+4,1)
    return 

   

def read_in_suite(path,mF, first_offset, n_components=6, record_stack_lenght=7):
    """
    path (str)                : path to the specific suite file
    mF (int)                  : specific fourier mode to read in the underlying file
    first_offset (int)        : position of the field, in the record stack, see below 
    n_components (int)        : number of components in fourier space, == 2*d with d the field dimension
    record_stack_lenght (int) : number of record (lines) written for each fourier mode
    
    #eg of suite_ns file structure (without LES and Multifluid options):
    
                    X : first line
             # Field stack begin
                    n : mode number = 0
                    u : field u(t)
                    u(m-1) : field u(t-dt)
                    p : field p(t)
                    p(m-1) : field p(t-dt)
                    icr_p : correction of field p(t)
                    incr_p(m-1) : correction of field p(t-dt)
            # Field stack end
                    n : mode number = 1
                    ...
                    
        eg : for u, first_offset=2, record_stack_lenght=7
        eg : for p, first_offset=4, record_stack_lenght=7
        eg : for les record_stack_lenght=8
        
    #eg of suite_maxwell file structure(without DNS and Multifluid) :
    
                    X : first line
             # Field stack begin
                    n : mode number = 0
                    H : field H(t)
                    H(m-1) : field H(t-dt)
                    B : field B(t)
                    B(m-1) : field B(t-dt)
                    phi : field phi(t)
                    phi(m-1) : field phi(t-dt)
            # Field stack end
                    n : mode number = 1
                    ...
                    
        eg : for H, first_offset=2, record_stack_lenght=7
        eg : for B, first_offset=4, record_stack_lenght=7
        
    returns :
    the raw field read in the suite file, on nodes points
    
    """
    with open(path,'rb') as file:
        # go to the given mF record
        skip_record(file,first_offset+mF*record_stack_lenght)
        
        # get record lenght
        record_length_bytes = file.read(4)
        record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
        
        #assuming double precision
        num_elements = record_length // 8
        n_nodes = num_elements//n_components
        field = np.fromfile(file,dtype=np.float64,count=num_elements).reshape(n_nodes,n_components,order='F')
    return field

def get_data_from_suites(sfem_par,I,mF_to_read,record_stack_lenght=7, opt_extension=''):
    """
    path_to_all_suites (str) : path to the directory where all the suites are stored
    path_to_mesh (str)       : path to the directory where Xmesh_jj are stored
    mF_to_read (int)         : specific fourier mode to read    
    S (int)                  : number of domains
    field_name_in_file (str) : field to read, must be in ["u","p","H","B"]
    record_stack_lenght(int) : see 'read_in_suite' function
    stack_domains (bool)     : if true the domains are stacked along a single array direction
     
    returns field :
    the underlying field, for mode mF_to_read.
    """


    # select between suite_ns and suite_maxwell
    if sfem_par.field == "u" or sfem_par.fielsd == 'un':
        # suite_kind="suite_ns"#f"{opt_extension}ns"
        mesh_kind = "vv"
        first_offset=2
        nb_components=6
        
    elif sfem_par.field == "un_m1":
        # suite_kind="suite_ns"#f"{opt_extension}ns"
        mesh_kind = "vv"
        first_offset=3
        nb_components=6

    elif sfem_par.field == "p" or sfem_par.field == 'pn':
        # suite_kind="suite_ns"
        mesh_kind = "pp"
        first_offset=4
        nb_components=2
        
    elif sfem_par.field == "pn_m1" :
        # suite_kind="suite_ns"
        mesh_kind = "pp"
        first_offset=5
        nb_components=2
    
    elif sfem_par.field == "incp" or sfem_par.field == 'incpn':
        # suite_kind="suite_ns"
        mesh_kind = "pp"
        first_offset=6
        nb_components=2
    
    elif sfem_par.field == "incpn_m1" :
        # suite_kind="suite_ns"
        mesh_kind = "pp"
        first_offset=7
        nb_components=2
    
    elif sfem_par.field == "H" :
        # suite_kind="suite_maxwell"
        mesh_kind = "H"
        first_offset=2
        nb_components=6
        
    elif sfem_par.field == "B": 
        # suite_kind="suite_maxwell"
        mesh_kind = "H"
        first_offset=4
        nb_components=6
        
    else:
        print("Field ",sfem_par.field," not found, or not implemented")
        return
    
    suite_names = []
    
    if I == -1:
        add_str = ''
    else:
        add_str = f'_I{I:03d}'

    nnodes = []
    
    for s in range(sfem_par.S):
        suite_name = sfem_par.path_suites+'/'+f'{sfem_par.name_suites}S{s:03d}{add_str}{sfem_par.mesh_ext}'
        data_compute_nn = read_in_suite(suite_name, 0, first_offset, nb_components, record_stack_lenght)
        nnodes.append(data_compute_nn.shape[0])
    nnodes = np.asarray(nnodes)

    field_out = np.zeros((nnodes.sum(), nb_components, mF_to_read.max()+1))
    n_inf = 0
    for s in range(sfem_par.S):
        n_sup = n_inf + nnodes[s]
        suite_name = sfem_par.path_suites+'/'+f'{sfem_par.name_suites}S{s:03d}{add_str}{sfem_par.mesh_ext}'
        for mF in mF_to_read:
            field_out[n_inf:n_sup, :, mF] = read_in_suite(suite_name, mF, first_offset, nb_components, record_stack_lenght)
        n_inf = n_sup

    return field_out

def get_suite(sfem_par,I,MF=None,record_stack_lenght=7, get_gauss_points=False,stack_domains=True, opt_time=False):
    
    if isinstance(I, int):
        I = [I]
    if MF is None:
        MF = np.arange(sfem_par.MF)
    elif isinstance(MF, int):
        print(f'WARNING: you chose to import only mF = {MF}, be sure it is defined/what you wanted to do')
        MF = [MF]
    if opt_time:
        if I[0] == -1:
            add_str = ''
        else:
            add_str = f'_I{I[0]:03d}'
        path_to_read = sfem_par.path_suites+'/'+f'{sfem_par.name_suites}S001{add_str}{sfem_par.mesh_ext}'
        with open(path_to_read,'rb') as file:
            record_length_bytes = file.read(4)
            record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
            num_elements = record_length // 8
            time = np.fromfile(file,dtype=np.float64,count=num_elements)[0]

    f_out = []
    for i in I:
        f_out.append(get_data_from_suites(sfem_par,i,MF,record_stack_lenght=record_stack_lenght))
    
    f_out = np.asarray(f_out)
    return time, f_out
