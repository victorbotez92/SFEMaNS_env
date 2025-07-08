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

def get_data_from_suites(par,I,mF_to_read,record_stack_lenght=7, get_gauss_points=True,stack_domains=True,opt_extension=''):
    """
    path_to_all_suites (str) : path to the directory where all the suites are stored
    path_to_mesh (str)       : path to the directory where Xmesh_jj are stored
    mF_to_read (int)         : specific fourier mode to read    
    S (int)                  : number of domains
    field_name_in_file (str) : field to read, must be in ["u","p","H","B"]
    record_stack_lenght(int) : see 'read_in_suite' function
    get_gauss_points (bool)  : if true the field is evaluated on gauss points
    stack_domains (bool)     : if true the domains are stacked along a single array direction
     
    returns field :
    the underlying field, for mode mF_to_read.
    """


    # select between suite_ns and suite_maxwell
    if par.field == "u":
        # suite_kind="suite_ns"#f"{opt_extension}ns"
        mesh_kind = "vv"
        first_offset=2
        nb_components=6
        
    elif par.field == "p" :
        # suite_kind="suite_ns"
        mesh_kind = "vv"
        first_offset=4
        nb_components=2
        
    elif par.field == "H" :
        # suite_kind="suite_maxwell"
        mesh_kind = "H"
        first_offset=2
        nb_components=6
        
    elif par.field == "B": 
        # suite_kind="suite_maxwell"
        mesh_kind = "H"
        first_offset=4
        nb_components=6
        
    else:
        print("Field",par.field,"not found, or not implemented")
        return
    
    suite_names = []
    
    if I == -1:
        add_str = ''
    else:
        add_str = f'_I{I:03d}'
    for s in range(par.S):
        # suite_names.append(f'{suite_kind}_S{s:03d}{add_str}{par.mesh_ext}')
        suite_names.append(f'{par.name_suites}S{s:03d}{add_str}{par.mesh_ext}')

    suite_files = [par.path_suites+'/'+elm for elm in suite_names]

    Nfile = len(suite_files)
    Nt = Nfile//par.S
    fields=[]
    for mF in mF_to_read:
        field_per_mF = []
        for path in suite_files:
            f = read_in_suite(path, mF, first_offset, nb_components, record_stack_lenght)
            field_per_mF.append(f)
            
        fields.append(field_per_mF) # eventually (MF S N (D a)
    TEMP=[]
    for mF in mF_to_read:
        TEMP_PER_MF = []
        for s in range(par.S):
            # TEMP_PER_MF.append( np.asarray(fields[mF, :, s*Nt:(s+1)*Nt])  )

            TEMP_PER_MF.append( np.asarray(fields[mF][:][s*Nt:(s+1)*Nt][0])  )
        TEMP.append(TEMP_PER_MF)
    if get_gauss_points :
        #get mesh info
        ME = [ read_mesh_info(par.path_to_mesh+f"/{mesh_kind}mesh_info_S{s:04d}.txt")[2] for s in range(par.S) ]  
        n_w = read_mesh_info(par.path_to_mesh+f"/{mesh_kind}mesh_info_S0000.txt")[0]
        l_G = read_mesh_info(par.path_to_mesh+f"/{mesh_kind}mesh_info_S0000.txt")[1]
        
        
        # get global connectivity, and test function weights
        mesh_ext= [f for f in listdir(par.path_to_mesh) if isfile(join(par.path_to_mesh, f)) if '.FEM' in f][0].split(".")[-2]+".FEM"
        mesh_jj = [ np.fromfile(par.path_to_mesh+f"/{mesh_kind}mesh_jj_S{s:04d}"+par.mesh_ext,dtype=np.int32).reshape(n_w,ME[s],order="F") 
                   for s in range(par.S) ] 
        mesh_ww = [ np.fromfile(par.path_to_mesh+f"/{mesh_kind}mesh_gauss_ww_S{s:04d}"+par.mesh_ext,dtype=np.float64).reshape(n_w,l_G,order="F") 
                   for s in range(par.S) ] 
        
        # node point to gauss points
        TEMP_gauss =[]   
        for mF in range(len(TEMP)):
            TEMP_gauss_per_mF = []
            for s in range(par.S):
                #arange field by triangle
                X = np.asarray( [TEMP[mF][s][mesh_jj[s][i]-1,:] for i in range(n_w)] )
                field_gauss = einsum(X,mesh_ww[s],'nw me d, nw l_G -> me l_G d ')
                TEMP_gauss_per_mF.append( rearrange(field_gauss, "me l_G d -> (me l_G) d") )
            TEMP_gauss.append(TEMP_gauss_per_mF)
        TEMP = TEMP_gauss
    
    if stack_domains:
        output=[]
        for TEMP_PER_MF in TEMP:
            # output_per_mF = []
            # for t in range(Nt):
                # output_per_mF.append(np.vstack([TEMP_PER_MF[s][t] for s in range(par.S)]))
            # output_per_mF.append(np.vstack([TEMP_PER_MF[s] for s in range(par.S)]))
            # output.append(output_per_mF)
            output.append(np.vstack([TEMP_PER_MF[s] for s in range(par.S)]))
        # data has now shape (MF,T,N_node_points_tot), domains are stacked : 0 then 1 then ... 
        return  np.asarray(output)
    else: 
        return TEMP 
    
def read_mesh_info(path):
    with open(path) as file:
        line = file.readlines()
        values = line[1::2][:4]
        n_w = int(values[0])
        l_G = int(values[1])
        me  = int(values[2])
        n_p = int(values[3])
    return n_w,l_G,me,n_p


def get_suite(par,I,MF=None,record_stack_lenght=7, get_gauss_points=True,stack_domains=True):
    if isinstance(I, int):
        I = [I]
    if MF is None:
        MF = np.arange(par.MF)
    elif isinstance(MF, int):
        print(f'WARNING: you chose to import only mF = {MF}, be sure it is defined')
        MF = [MF]
    
    f_out = []
    for i in I:
        f = get_data_from_suites(par,i,MF,record_stack_lenght=record_stack_lenght, get_gauss_points=get_gauss_points,
        stack_domains=stack_domains)
        f_out.append(rearrange(f,'MF N D_a -> N D_a MF'))

    f_out = np.array(f_out)
    return f_out
