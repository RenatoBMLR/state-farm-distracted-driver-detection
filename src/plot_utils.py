import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from src.imgnet_utils import denormalize



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
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.xticks(rotation=90)
    plt.title("Classes Distribution", fontsize=15)
    plt.show()
    path2save = './figures/distribution_classes.png'
    fig.savefig(path2save)
    
  