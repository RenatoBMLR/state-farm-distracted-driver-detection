import numpy as np
import random
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.imgnet_utils import denormalize
import torchvision
import torch
from torch.autograd import Variable
from matplotlib.ticker import MaxNLocator


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
    fig.savefig(path2save)
    
    

def rename_metrics_keys(metrics):
    
    #renaming dictionary keys
    metrics['train']['acc'] = metrics['train'].pop(0)
    metrics['train']['losses'] = metrics['train'].pop(1)

    metrics['valid']['acc'] = metrics['valid'].pop(0)
    metrics['valid']['losses'] = metrics['valid'].pop(1)
    return metrics


def plot_metrics(path2metrics, path2save = []): 

    metrics = pd.read_csv(path2metrics).to_dict()
    metrics = rename_metrics_keys(metrics)

    metrics_map = {'losses': 'Loss', 'acc': 'Acuracy'}

    metrics_eval_nb = len(metrics['train'].keys())
    
    fig = plt.figure(figsize=(10*metrics_eval_nb,5))

    count = 1
    for metric in metrics['train'].keys():
        ax = fig.add_subplot(1,metrics_eval_nb, count)
        ax.plot(range(len(ast.literal_eval(metrics['train'][metric]))), ast.literal_eval(metrics['train'][metric]), 'o-b', label = 'train')
        ax.plot(range(len(ast.literal_eval(metrics['train'][metric]))), ast.literal_eval(metrics['valid'][metric]), 'o-r', label = 'valid')
        count += 1
        
        ax.set_xlabel('Epochs', fontsize = 12)
        ax.set_ylabel(metrics_map[metric], fontsize = 12)
        ax.set_title(metrics_map[metric] + " during the model's training", fontsize = 16)
        ax.grid('on')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        
        if len(path2save) !=0:
            fig.savefig(path2save)

        
def visualize_predictions(dsets, lst, results, path2save = []):
    
    
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
        plt.title('{}:{}:{:0.3f}'.format('c'+str(output),  'c'+str(results['pred'][lst[j]]), results['probas'][lst[j]]))    
        plt.axis('off')
    if len(path2save) !=0:
        fig.savefig(path2save)    

def plot_confusion(results):
    mc = np.array(pd.crosstab(results['pred'], results['true']))
    path2save = './figures/distribution_classes.png'
    path2save = './figures/distribution_classes.png'
    plt.imshow(mc/mc.sum(axis=1), cmap = 'jet')
    plt.colorbar()
    plt.axis('off')
    
def plot_cm_train_valid(result_train,result_valid, path2save = []):    
    fig = plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plot_confusion(result_train)
    plt.title('Train dataset')
    plt.subplot(1,2,2)
    plot_confusion(result_valid)
    plt.title('Valid dataset')
    if (len(path2save)) != 0:
        fig.savefig(path2save)

    
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


def models_comparasion(models, m = 'acc', path2save = []):

    metrics_dict = {}
    eval = {}
    train_lst = []
    valid_lst = []


    metrics_map = {'losses': 'Loss', 'acc': 'Acuracy'}
    eval = {'train': [], 'valid': []}
    for model in models:

        path2metrics = './metrics/metrics_'+ model+'.csv'
        metrics = pd.read_csv(path2metrics).to_dict()
        metrics = rename_metrics_keys(metrics)
        metrics_dict[model] = metrics

        for daset in metrics_dict[model].keys():
            if m == 'losses':
                x=np.min(ast.literal_eval(metrics_dict[model][daset][m]))
            else:
                x=np.max(ast.literal_eval(metrics_dict[model][daset][m]))
            eval[daset].append(x)

    ind = np.arange(len(models))  # the x locations for the groups
    width = 0.15       # the width of the bars

    print(metrics_map[m] + ' Training set: {}'.format(eval['train']) )
    print(metrics_map[m] + ' Valid set: {}'.format(eval['valid']) )
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, eval['train'], width, color='r')
    rects2 = ax.bar(ind + width, eval['valid'], width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel(m)
    ax.set_title(metrics_map[m] + ' obtained during the training')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple(models))

    ax.legend((rects1[0], rects2[0]), ('train', 'valid'))

    if (len(path2save)) !=0:
        fig.savefig(path2save)
