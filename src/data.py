#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:06:00 2017

@author: renatobottermaiolopesrodrigues
"""


from PIL import Image
import os
import nuumpy as np
import pandas as pd
import glob

class get_data():
    def __init__(self, path2data, path2labels, img_rows =224, img_cols=224 ):
        self.path2data = path2data
        self.path2labels = path2labels
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        
        self.trainLabels = pd.read_csv(self.path2labels)

        immatrix = []
        imlabel = []
        
        files = glob.glob (self.path2data + "*.jpeg")
        for myFile in files:    
            base = os.path.basename(myFile)
            fileName = os.path.splitext(base)[0]
            imlabel.append(self.trainLabels.loc[self.trainLabels.image==fileName, 'level'].values[0])
            im = Image.open(myFile)
            img = im.resize((self.img_rows,self.img_cols))
            immatrix.append(np.array(img))    
             
        self.immatrix = np.asarray(immatrix)
        self.imlabel = np.asarray(imlabel)




        
    
    
    
        
        
