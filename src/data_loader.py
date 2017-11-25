
import torch
from torch.utils.data import DataLoader



def _create_dataLoader(dsets, batch_size,  pin_memory=False, use_shuffle=False):
    '''Arguments: 
        dset: Dictionary with datasets for train, validation and test;
        
        batch_size: Size of batch **If you are using nn.DataParalel this should be an multiple of GPUS available on your system**;
       
        num_workers: Number of multiprocessing threads on CPU only, beware to initialize nn.multiprocessing with forkserver or spwanw;
        
        use_gpu: Flag to use one gpu only;
        
        use_DataParalel: Flag to use multiple GPU's;
        
        shuffle: Flag to either shuffle data or not;
    '''
    dset_loaders = {}
   
    shuffle = False
    for key in dsets.keys():
        if use_shuffle:
            if key != 'test':
                shuffle = True
            else:
                shuffle = False
        dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle = shuffle)
            
    return dset_loaders
    
    

