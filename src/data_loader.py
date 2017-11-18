
import torch
from torch.utils.data import DataLoader



def _create_dataLoader(dsets, batch_size, num_workers= None, use_gpu=None,use_DataParalel=None, shuffle=True):
    '''Arguments: 
        dset: Dictionary with datasets for train, validation and test;
        
        batch_size: Size of batch **If you are using nn.DataParalel this should be an multiple of GPUS available on your system**;
       
        num_workers: Number of multiprocessing threads on CPU only, beware to initialize nn.multiprocessing with forkserver or spwanw;
        
        use_gpu: Flag to use one gpu only;
        
        use_DataParalel: Flag to use multiple GPU's;
        
        shuffle: Flag to either shuffle data or not;
    '''
    dset_loaders = {}
   
    if use_gpu:

        for key in dsets.keys():
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, 
                                           shuffle=shuffle, num_workers= 1, pin_memory=use_gpu)
    if use_DataParalel:
        for key in dsets.keys():
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, 
                                           shuffle=shuffle)

    if use_gpu == False and use_DataParalel== False:
        for key in dsets.keys():
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, 
                                           shuffle=shuffle, num_workers= num_workers, pin_memory=False)            
            
    return dset_loaders
    
    

