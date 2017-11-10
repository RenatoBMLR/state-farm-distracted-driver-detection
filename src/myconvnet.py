#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 9 20:06:00 2017

@author: renatobottermaiolopesrodrigues
"""



class myconvNet(nn.Module):
    def __init__(self, image_size=(3,50,50)):
        super(myconvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        
        
        feature_size = self._get_conv_output(image_size)
        
        self.dense1 = nn.Linear(feature_size, 250)
        self.dense2 = nn.Linear(250, 125)
        self.dense3 = nn.Linear(125, nb_classes)
        
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(tc.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
       
        return x
    
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense3(x))
        return F.softmax(x)


myconvNet = myconvNet()
if use_gpu:
    myconvNet = myconvNet.cuda()

