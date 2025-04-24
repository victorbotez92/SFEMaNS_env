import numpy as np

from einops import rearrange,repeat
import sys,os,array
from pathlib import Path
from get_par import SFEMaNS_par

#================Get meshes
def get_mesh(par):
    R = np.hstack([np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext) for s in range(par.S)]).reshape(-1)
    Z = np.hstack([np.fromfile(par.path_to_mesh+f"/{par.mesh_type}zz_S{s:04d}"+par.mesh_ext) for s in range(par.S)]).reshape(-1)
    W = np.hstack([np.fromfile(par.path_to_mesh+f"/{par.mesh_type}weight_S{s:04d}"+par.mesh_ext) for s in range(par.S)]).reshape(-1)
    return R,Z,W

#==============Elementary functions for getting data
def get_size(path):
    with open(path, 'rb') as fin:
        n = os.fstat(fin.fileno()).st_size // 8
    return n

def get_file(path,n):
    data = array.array('d')
    with open(path, 'rb') as fin:
        data.fromfile(fin, n)
    return data

#===============Functions to get data from stbp
def get_phys(par,I):# output shape is (D theta N)
    N = [len(np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext)) for s in range(par.S)]
    N_tot = np.sum(np.array(N))
    N_slice=np.cumsum(np.array(N))

    data = np.zeros(shape=(par.D,2*par.MF-1,N_tot))

    for s in range(par.S):
        n = N[s]*(2*par.MF-1)
        for d in range(par.D):
            if par.D > 1:
                path=par.path_to_suite+"/phys_{f}{i}_S{s:04d}_I{m:04d}".format(f=par.field,i=d+1,s=s,m=I)+par.mesh_ext
            elif par.D == 1:
                path=par.path_to_suite+"/phys_{f}_S{s:04d}_I{m:04d}".format(f=par.field,s=s,m=I)+par.mesh_ext

            new_data = np.array(get_file(path,n))
            new_data = rearrange(new_data,'(theta n) -> theta n', theta=2*par.MF-1)

            if s==0:
                data[d,:,:N_slice[s]]=np.copy(new_data[:,:])
            else:
                data[d,:,N_slice[s-1]:N_slice[s]]=np.copy(new_data[:,:])
    return data

def get_fourier(par,I,MF=[],fourier_type=["c","s"]):# output shape is (D (MF*a) N)
    N = [len(np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext)) for s in range(par.S)]
    N_tot = np.sum(np.array(N))
    N_slice=np.cumsum(np.array(N))

    if MF == []:
        MF = np.arange(par.MF)
    n_mF = len(MF)

    data = np.zeros(shape=(par.D,n_mF*2,N_tot))

    for s in range(par.S):
        n = N[s]*par.MF
        for d in range(par.D):
            for a,axis in enumerate(fourier_type):
                array_a = 2*np.arange(len(MF))+a
                if par.D > 1:
                    path=par.path_to_suite+"/fourier_{f}{i}{ax}_S{s:04d}_I{m:04d}".format(f=par.field,i=d+1,ax=axis,s=s,m=I)+par.mesh_ext
                elif par.D == 1:
                    path=par.path_to_suite+"/fourier_{f}{ax}_S{s:04d}_I{m:04d}".format(f=par.field,ax=axis,s=s,m=I)+par.mesh_ext

                new_data = np.array(get_file(path,n))
                new_data = rearrange(new_data,'(MF n) -> MF n', MF=par.MF)
                if s==0:
                    data[d,array_a,:N_slice[s]]=np.copy(new_data[MF,:])
                else:
                    data[d,array_a,N_slice[s-1]:N_slice[s]]=np.copy(new_data[MF,:])
    return data

def get_fourier_per_mode(par,mF,T=-1,fourier_type=["c","s"]):# output shape is (T D a N)
    N = [len(np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext)) for s in range(par.S)]
    N_tot = np.sum(np.array(N))
    N_slice=np.cumsum(np.array(N))
    if T == -1:
        T = par.I
    data = np.zeros(shape=(T,par.D,2,N_tot))
    for s in range(par.S):
        n = N[s]*par.I
        for d in range(par.D):
            for a,axis in enumerate(fourier_type):
                if par.D > 1:
                    path=par.path_to_suite+"/fourier_{f}{i}{ax}_S{s:04d}_F{m:04d}".format(f=par.field,i=d+1,ax=axis,s=s,m=mF)+par.mesh_ext
                elif par.D == 1:
                    path=par.path_to_suite+"/fourier_{f}{ax}_S{s:04d}_F{m:04d}".format(f=par.field,ax=axis,s=s,m=mF)+par.mesh_ext

                new_data = np.array(get_file(path,n))
                new_data = rearrange(new_data,'(T n) -> T n', T=T)#new_data.reshape(T,len(new_data)//T)
                if s==0:
                    data[:,d,a,:N_slice[s]]=np.copy(new_data[:T,:])
                else:
                    data[:,d,a,N_slice[s-1]:N_slice[s]]=np.copy(new_data[:T,:])
    return data


