import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

#********************************************           Predicting Features       *********************************************** 

def predict(dset_loaders, model,use_gpu=False):
    '''
    This function returns the features of the features extractor model which have been trained on ImageNet dataset.
    Arguments : 
        dset_loaders: Dataset Loader;
        model : Pretrained model which will act as a feature extractor;
        use_gpu: Flag to use only one GPU;
        device_id: Id of which GPU will be used;
    
    
    '''
    predictions = []
    labels_lst = []
    ii_n = len(dset_loaders)

    for i, (inputs, labels) in enumerate(dset_loaders):
        
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        inputs = Variable(inputs)
        predictions.append(model(inputs).data)
        labels_lst.append(labels)

        print('\rpredict: {}/{}'.format(i, ii_n - 1), end='')
    print(' ok')
    if len(predictions) > 0:
        return {'pred': torch.cat(predictions, 0), 'true':torch.cat(labels_lst, 0) }




def getPrediction(result):
    _, predicted = torch.max(result['pred'], 1)
    result['pred'] = predicted.cpu().numpy()
    result['true'] = result['true'].cpu().numpy()
    return result 
    
def features_saving(path2data,convOutput,use_resnet=False,use_inception=False,use_denseNet=False):
    """ This function saves the features extracted by the model as 'npz' archive of numpy arrays. 
     Arguments:
     path2data: Path to save the features.
     convOutput: Dictinary with features extracted (tensors)
     use_resnet: Flag to chose the name of the archive corresponding to model used;
     use_inception: Flag to chose the name of the archive corresponding to model used;
     use_denseNet: Flag to chose the name of the archive corresponding to model used;
    
    """
    data ={'true':convOutput['true'].numpy(),
           'pred':convOutput['pred'].numpy()}
    if use_resnet:
        print('Saving ResNet features') 
        np.savez(path2data+"ResNetFeatures.npz",**data)
    elif use_inception:
        print('Saving Inception features')
        np.savez(path2data+"InceptionFeatures.npz",**data)
    elif use_denseNet:
        print('Saving DenseNet features')    
        np.savez(path2data+"DenseNetFeatures.npz",**data)    
    
    