"""
    This file reads in hdf5 files using Jorrit Schaap's interfacing framework: lofarReadSnippet
    Misha Mesarcik 2020
"""

import os
import numpy as np
from h5py import File
from numpy import random
from random import shuffle
from numpy import sqrt,angle

from cluster_tools import lofarReadSnippet  

from config import config

def get_files(filter =None):
    """
        Gets all h5 files in basedir specified by config
        Acts as a generator for the training process
    """
    while True:
        files_names = [f for f in os.listdir(config['path']) if os.path.isfile(os.path.join(config['path'], f))]
        files_names.remove('.DS_Store')
        h5_files =  [ os.path.join(config['path'], f) for f in files_names]
        shuffle(h5_files)

        if filter == 'shape':
            h5_files_filtered = [] 
            for h5_file in h5_files:
                temp = File(h5_file)
                sap_0 = list(temp['measurement']['saps'].keys())[0]
                s = (temp['measurement']['saps'][str(sap_0)]['visibilities'].shape)
                if (s[1] == 59 and s[2] == 16) or ( s[1] == 90 and s[2] == 8) :
                    h5_files_filtered.append(h5_file) 
            h5_files = h5_files_filtered
        for h5_file in h5_files:
            yield h5_file

def get_random_sap(h5_file):
    """
        Gete a random sap of a specified h5 file
        h5_file (h5py.File): The file to be prcoessed
    """
    saps = list(h5_file['saps'].keys())
    sap = random.choice(saps)  
    return sap 

def get_station_mask(h5_file,sap_key,filter= 'core'):
    """
        Gets a mask fo stations to be selected
        h5_file (h5py.File): The file to be preocessed
        sap_key (int): The sap number to be processed
        core (bool): Specifies a filter for core stations
    """
    mask = []
    for baseline in h5_file['saps'][sap_key]['baselines']:
        flag = False 
        if filter == 'core':
            for station in baseline:
                if "CS" not in station:
                    flag = True 
                    break
        if filter == 'Cross':
            flag = (baseline[0] == baseline[1]) 

        if filter == 'None':
            flag = True
        mask.append(flag)
    return mask

def get_station_names(h5_file,sap_key):
    """
        Gets the stations names for a particular h5_file 
        h5_file (h5py.File): The file to be preocessed
        sap_key (int): The sap number to be processed
    """
    names = [baseline[0].decode('utf-8') + '-' + baseline[1].decode('utf-8')
            for baseline in h5_file['saps'][sap_key]['baselines']]
    return np.array(names)

def restructure_cube(cube,sap):
    """
        Peforms the orginal reformatting
    """
    cube = cube['saps'][sap]['visibilities'].transpose([0, 2, 1, 3])
    cube = np.concatenate((cube.real, cube.imag), axis=3)

    return cube

def get_random_batches(cube):
    idx = random.randint(cube.shape[0],size = config['batch_size'])
    return cube[idx,:]

def get_processed_cube(file_name, core = 'None',verbose = True):
    """
        from a  random file we get random sap and station mask and returns a cube
    """
    if verbose: print('\n Training on: {}'.format(file_name))

    h5_file = lofarReadSnippet.read_hypercube(file_name,visibilities_in_dB=True,
                                              read_visibilities=True,read_flagging=False)
    sap =  get_random_sap(h5_file)
    mask = get_station_mask(h5_file,sap,filter=core)
    cube = restructure_cube(h5_file,sap)
    core_vis = cube[mask]
    
    return core_vis

def get_cube(file_name, filter = 'None',verbose = True,batches=False,names = False):
    """
        from a  random file we get random sap and station mask and returns a cube
    """
    if verbose: print('\n Training on: {}'.format(file_name))

    h5_file = lofarReadSnippet.read_hypercube(file_name,visibilities_in_dB=True,
                                              read_visibilities=True,read_flagging=False)
    sap =  get_random_sap(h5_file)
    station_names = get_station_names(h5_file,sap)
    mask = get_station_mask(h5_file,sap,filter=filter)
    cube = restructure_cube(h5_file,sap)
    core_vis = cube[mask]
   
    if names: (core_vis,station_names[mask])
    if batches: return get_random_batches(core_vis)
    if not names and not batches: return core_vis

