import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.imgnet_utils import denormalize
import torchvision
import torch
from torch.autograd import Variable





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




def plot_classes(dset_loaders, path2save = './figures/data.png'):
    ''' This function shows some classes of the dataset passed as argument
    '''
    # Get a batch of training data
    inputs, cls = next(iter(dset_loaders))
    print(inputs[8].shape, cls.shape)
    fig = plt.figure(figsize=(15,6))
    for i, j in enumerate(range(0,10)):
        fig.add_subplot(2,5, i+1)
        img = denormalize(inputs.numpy()[j])
        img = np.clip(img, 0, 1.0)
        plt.imshow(img)
        plt.title('{}'.format(labels['c'+str(cls[j])]))
        plt.axis('off')
    fig.savefig(path2save)
    
    
    
def plot_distribution(img):
    ''' This function plot the RGB distribution of a given image
    '''
    if img.shape[2] != 3:
        img = img.transpose((1, 2, 0))

    color_lst = ['red', 'green', 'blue']
    for i in range(0, img.shape[2]):
        c1=img[:,:,i].reshape(-1)
        plt.hist(c1, 50, facecolor=color_lst[i], label = color_lst[i])
    plt.legend()
    plt.grid(True)    
    
    
def statistical_analysis_image(dset_loaders, path2save = './figures/distribution.png'):

    ''' This function shows the RGB distribution of an image chosen randomly from a given dataset
        before and after the normalization process
    '''
    
    fig = plt.figure(figsize=(15,6))
    inputs, cls = next(iter(dset_loaders))
    rand_idx = random.randrange(0, len(inputs))
    img = inputs.numpy()[rand_idx]
    img_denorm = denormalize(img)
    plt.subplot(2,2,2)
    plot_distribution(img_denorm)
    plt.title('Image RGB after denormalization')
    plt.subplot(2,2,4)
    plot_distribution(img)
    plt.title('Image RGB normalization')
    plt.subplot(1,2,1)
    img_denorm = np.clip(img_denorm, 0, 1.0)
    plt.imshow(img_denorm)
    plt.title('{}'.format(labels['c'+str(cls[rand_idx])]))
    plt.axis('off')
    fig.savefig(path2save)    
    

def classDistribution(dataset, path2save= './figures/class_distribution.png'): 
    ''' This function plot the class distribution in the given dataset
    '''
    
    class_str = []
    for item in dataset.y:
        class_str.append(labels['c'+str(item)])  
    fig = plt.figure(figsize=(8,10))
    sns.countplot(y=class_str, palette="Greens_d");
    plt.ylabel("Classes", fontsize=12)
    plt.xlabel('Count', fontsize=12)
    plt.xticks(rotation=90)
    plt.title("Classes Distribution", fontsize=15)
    plt.show()
    path2save = './figures/distribution_classes.png'
    fig.savefig(path2save)
    
def plot_metrics(trainer): 

    path2save = './figures/results_metrics.png'

    fig = plt.figure(figsize=(10,5))
    metrics_map = {'losses': 'Loss', 'acc': 'Acuracy'}

    metrics_eval_nb = len(trainer.metrics['train'].keys())
    count = 1
    for metric in trainer.metrics['train'].keys():
        plt.subplot(1,metrics_eval_nb, count)
        plt.plot(trainer.metrics['train'][metric], 'o-b', label = 'train')
        plt.plot(trainer.metrics['valid'][metric], 'o-r', label = 'valid')
        count += 1
        plt.xlabel('Epochs', fontsize = 12)
        plt.ylabel(metrics_map[metric], fontsize = 12)
        plt.title(metrics_map[metric] + " during the model's training", fontsize = 16)
        plt.grid('on')
        plt.legend()
    fig.savefig(path2save)

        
def visualize_predictions(dsets, results, path2save = [], correct_pred = True):
    
    if correct_pred == True:
        lst = np.where(results['true'] == results['pred'])[0]
    else:
        lst = np.where(results['true'] != results['pred'])[0]
    
    maxSubPlot = 4
    if len(lst)<4:
        maxSubPlot = len(lst)
    fig = plt.figure(figsize=(15,6))
    for i, j in enumerate(range(0,maxSubPlot)):
        fig.add_subplot(1, maxSubPlot, i+1)
        (inputs, output) = dsets[lst[j]]
        img = denormalize(inputs.numpy())
        img = np.clip(img, 0, 1.0)
        plt.imshow(img)
        #plt.title('{0} / {1}'.format(labels['c'+str(output)],  labels[('c'+str(results['pred'][lst[j]]))]))    
        plt.title('{0} / {1}'.format(('c'+str(output)),  ('c'+str(results['pred'][lst[j]]))))    
        plt.axis('off')
    if len(path2save) !=0:
        fig.savefig(path2save)    

def plot_confusion(results):
    mc = np.array(pd.crosstab(results['pred'], results['true']))
    plt.imshow(mc/mc.sum(axis=1))
    plt.colorbar()
    plt.axis('off')
    
def plot_cm_train_valid(result_train,result_valid):    
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plot_confusion(result_train)
    plt.title('Train dataset')
    plt.subplot(1,2,2)
    plot_confusion(result_valid)
    plt.title('Valid dataset')

    
def plot_layers_weight(dsets,img_width, img_height, conv_model,use_gpu, ncols = 8, H = 14, W=30):


    rand_idx = random.randrange(0, len(dsets['train']))
    input, _ = dsets['train'][rand_idx]
    input = input.view(1, 3, img_width, img_height)

    if use_gpu:
        x = Variable(input.cuda())
    else:
        x = Variable(input)


    for name, layer in conv_model.named_children():
        x = layer(x)
        grid = torchvision.utils.make_grid(torch.transpose(x.data, 0, 1), normalize=True, 
                                           pad_value=1.0, padding=1).cpu().numpy()

        if name == 'max_pool':
            H /= 3/2
            W /= 3/2
        fig = plt.figure(figsize=(H,W))
        plt.imshow(grid.transpose((1,2,0)))
        plt.title(name)
        plt.axis('off')
        plt.show()    