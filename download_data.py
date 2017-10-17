#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:50:52 2017

@author: renatobottermaiolopesrodrigues
"""

import shutil
import requests
import sys 

login_url = 'https://www.kaggle.com/account/login'
download_url = 'https://www.kaggle.com/c/diabetic-retinopathy-detection/download/trainLabels.csv.zip'
filename = download_url.split('/')[-1]
login_data = {'UserName': sys.argv[1], 
              'Password': sys.argv[2]}

with requests.session() as s, open(filename, 'w') as f:
    s.post(login_url, data=login_data)                  # login
    response = s.get(download_url, stream=True)         # send download request
    shutil.copyfileobj(response.raw, f)                 # save response to file