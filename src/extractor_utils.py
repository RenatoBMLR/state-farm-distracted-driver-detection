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
            
        inputs = Variable(inputs,volatile=True)
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




def features_saving(path2data,convOutput):
    """ This function saves the features extracted by the model as 'npz' archive of numpy arrays. 
     Arguments:
     path2data: Path to save the features.
     convOutpu: Dictinary with features extracted (tensors)
    
    """

    
    for key in convOutput.keys():
        if convOutput[key][0].is_cuda: 

            data ={'pred':convOutput[key][0].cpu().numpy(),
                   'true':convOutput[key][1].cpu().numpy()}
        else:

            data ={'pred':convOutput[key][0].numpy(),
                   'true':convOutput[key][1].numpy()}
        
        print('\nSaving '+convOutput[key][2]+' '+ key+' features') 
        np.savez(path2data+key+"/"+convOutput[key][2]+"Features.npz",**data)
        print('Saved in:'+path2data+key+"/"+convOutput[key][2]+"Features.npz")


def features_loading(path2data,model_name,use_gpu=False):
    l = ['train','valid','test']
    data ={'train':(),
           'valid':(),
           'test':(),}
    
    print("Loaded features with shapes: \n")
    for i in l:
        npzfile = np.load(path2data+i+"/"+model_name+"Features.npz")

        if use_gpu:
            data[i]= (torch.from_numpy(npzfile['pred']).cuda(),torch.from_numpy(npzfile['true']).cuda())

        else:
            data[i]=  (torch.from_numpy(npzfile['pred']), torch.from_numpy(npzfile['true']))

            print('\n'+i+':')
            print("pred {}, true {}".format(data[i][0].shape,data[i][1].shape))

    return data

