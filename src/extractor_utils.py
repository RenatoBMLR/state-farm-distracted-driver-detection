import os
import numpy as np
import numpy.random as nr
import copy
import time
import datetime

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.module import _addindent


import lib.pytorch_trainer as ptt

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




def save_prediction(path2data,convOutput):
    """ This function saves the features extracted by the model as 'npz' archive of numpy arrays. 
     Arguments:
     path2data: Path to save the features.
     convOutpu: Dictinary with features extracted (tensors)
    
    """

    
    for key in convOutput.keys():
        if convOutput[key][0].is_cuda: 

            data ={'true':convOutput[key][0].cpu().numpy(),
                   'pred':convOutput[key][1].cpu().numpy()}
        else:

            data ={'true':convOutput[key][0].numpy(),
                   'pred':convOutput[key][1].numpy()}
        if not os.path.isdir(path2data + key):
            os.mkdir(path2data + key)
        
        print('\nSaving '+convOutput[key][2]+' '+ key) 
        np.savez(path2data+key+"/"+convOutput[key][2]+".npz",**data)
        print('Saved in:'+path2data+key+"/"+convOutput[key][2]+".npz")
        
        
def load_prediction(path2data,model_name,use_gpu=False):
    l = ['train','valid','test']
    data ={'train':(),
           'valid':(),
           'test':(),}
    
    print("Loaded features with shapes: \n")
    for i in l:
        npzfile = np.load(path2data+i+"/"+model_name+".npz")

        if use_gpu:
            data[i]= (torch.from_numpy(npzfile['true']).cuda(),torch.from_numpy(npzfile['pred']).cuda())

        else:
            data[i]=  (torch.from_numpy(npzfile['true']), torch.from_numpy(npzfile['pred']))

            print('\n'+i+':')
            print("pred {}, true {}".format(data[i][0].shape,data[i][1].shape))

    return data


def RandomSearch(param,args,num_epochs,path2saveModel,dset_loaders_convnet,MAX_IT,verbose):
    '''
    This function searchs for best lr and weight_decay parameters of a given model
     Args:
        param: Dictionary with parameters as follow: 
                    params = { 'model' : Instance of model, 
                               'criterion': Loss function used,  
                               'optimizer': instance of optimizer, 
                               'callbacks': list of callbacks }

        args: Dictionary with minimum and max values of lr and weight decay. Example: 
                    args= {'lr':[1e-5,1e-2],
                           'weight_decay':[1e-8,1e-3] }
                           
        num_epochs: Number of epochs to train in each iteration
        path2saveModel: Path to load model weights of best epoch in each itearation 
        dset_loaders_convnet: Data loaders with train and valid tensors
        MAX_IT: Number of maximum interation
        verbose: 0 None, 1 little, 2 full verbose
        
    returns: Dictionary with: history: val_loss and train loss,
                              Best result,
                              Best parameters,
                              Best Model,
                              Best Trainer
    '''


    results=[]

    l={}
    
    for name, value in args.items():
        l[name] = nr.permutation(nr.uniform(low= value[0],
                                                 high=value[1],size=10*MAX_IT))
   
    
    # Saving natural state of model
    natural_state = param['model'].state_dict()

    # Copy search space to get best_parameters later on
    searchSpace = copy.deepcopy(l)

    # Adding PrintCallback
    if verbose ==2: 
        param['callbacks'].append(ptt.PrintCallback())


    
    for i in range(MAX_IT):
        # Getting start time 
        start_time = time.time()
        
        if verbose >0:    
            print('Iteration {}/{}'.format(i,MAX_IT))
                
        # Reseting model weights and adjusting optimizer parameters
        param['model'].load_state_dict(natural_state)
        param['optimizer'] = optim.Adam(param['model'].parameters(), 
                                        lr = l['lr'][i],
                                        weight_decay = l['weight_decay'][i]) 
        
        
        # Trainning model 
        trainer = ptt.DeepNetTrainer(use_gpu=True,**param)
       
        trainer.fit_loader(num_epochs,
                           dset_loaders_convnet['train'],
                           dset_loaders_convnet['valid'])

        # Taking best results
        trainer.load_state(path2saveModel)

        train_eval = trainer.evaluate_loader(dset_loaders_convnet['train'],verbose=verbose)
        valid_eval = trainer.evaluate_loader(dset_loaders_convnet['valid'],verbose=verbose)

        # Deep copy of trainer
        obj = copy.deepcopy(trainer)

        results.append([valid_eval['losses'],
                        train_eval['losses'],
                        obj])

        if verbose >= 1:    
            print('train_loss: {}, val_loss {}'.format(train_eval['losses'],
                                                       valid_eval['losses']))
            print ('Execution time :{} s'.format(time.time() - start_time))
            if verbose == 2 :
                print('lr: {}, weight_decay: {}'.format(l['lr'][i],
                                                       l['weight_decay'][i]))

                
        # Removing parameters used
        l['lr'] = np.delete(l['lr'],i)
        l['weight_decay'] = np.delete(l['weight_decay'],i)

        

    # Getting best result
    best_result = min(results)

    # Getting best parameters
    index = results.index(best_result)    
    best_parameters={'lr': searchSpace['lr'][index],
                     'weight_decay': searchSpace['weight_decay'][index]}

    # Getting best model
    best_trainer = results[index][2]
    best_model = best_trainer.model

    if  os.path.isfile(path2saveModel+'.model'):
        os.unlink(path2saveModel + '.model')
    torch.save(best_model.state_dict(), path2saveModel + '.model')
    
    
    # History of valid_loss and train_loss
    history = []
    history = [x[:2] for x in results]
    
    output={'history':history,
            'best_result':best_result,
            'best_parameters':best_parameters,
            'best_model':best_model,
            'best_trainer':best_trainer,
            }
    
    # Saving history for further analysis 
    np.savez(path2saveModel+'History',*output['history'])
    
    return output





def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

