import os
import re
import glob
import numpy as np
import numpy.random as nr

from torch.utils.data import Dataset
from PIL import Image


# Class labels
labels = {  'c0' : 'safe driving', 
            'c1' : 'texting - right', 
            'c2' : 'talking on the phone - right', 
            'c3' : 'texting - left', 
            'c4' : 'talking on the phone - left', 
            'c5' : 'operating the radio', 
            'c6' : 'drinking', 
            'c7' : 'reaching behind', 
            'c8' : 'hair and makeup', 
            'c9' : 'talking to passenger'}


class KaggleSafeDriverDataset(Dataset):
    """
    Arguments:
        path: Path to data (train or test) 
        use_only: Percentage of total data that will be used.
        transforms: PIL transforms to be perfomed on each item of get_item method
        is_test: Test data (boolean)
        is_val: Validation data (boolean)
        val_size: Size of validation data 
        
        **** The indices of Validation and Train dataset are shuffled****
        
    """

    def __init__(self, path, use_only =1.0, transforms=None, \
                 is_test=False,is_val=False,val_size=0.2):
    
        self.transform = transforms        
        self.is_test = is_test
        if self.is_test:

            X_test    = []
            path  = os.path.join(path, '*.jpg')
            files = glob.glob(path)
            X_test.extend(files)  

            length = len(X_test)
            only = int(use_only * length)
            self.X = X_test[:only]
            
            self.y = np.zeros([len(self.X), 1]) # In order to create the Dataloader of test data it has to have a y coordenate.
        
        else:

            X_train= []
            y_train = []
            for i, label in enumerate(labels):
                path_folder = os.path.join(path, str(label), '*.jpg')
                files = glob.glob(path_folder) 
                X_train.extend(files)
                y_train.extend([int(label[-1])]*len(files))

            length = len(X_train)
            indices = np.array((range(0,length)))
            
            nr.seed(4572)
            ind = nr.permutation(indices)
            
            
            length = ind.shape[0]
            only = int(use_only * length)
            
            ind = ind[:only]
            length = ind.shape[0]
                 
            split = int(val_size * length)
            
            if is_val:
                self.X = [X_train[i] for i in ind[:split]]
                self.y = [y_train[i] for i in ind[:split]]
 
            else:
                self.X = [X_train[i] for i in ind[split:]]
                self.y = [y_train[i] for i in ind[split:]]
    
    def __getitem__(self, index):
        path = self.X[index]
        label = self.y[index]
        with open(path, 'rb') as f:
            flbase = os.path.basename(path)
            if self.is_test:
                id_img = re.findall('\d+', flbase)
                label = int(id_img[0])
            with Image.open(f) as img:
                 image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.X)
    