import torchvision
import torch
from torch.autograd import Variable

nb_out = 10;


#ResNet

class MyResNetConv(torchvision.models.ResNet):
    def __init__(self, fixed_extractor = True):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(torch.utils.model_zoo.load_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        
        del self.fc
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MyResNetDens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=4096, out_features=512)
        self.dens2 = torch.nn.Linear(in_features=512, out_features=128)
        self.dens3 = torch.nn.Linear(in_features=128, out_features=nb_out)
    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = self.dens2(x)
        x = torch.nn.functional.selu(x)
        x = self.dens3(x)

        return x

class MyResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyResNetConv()
        self.mrnd = MyResNetDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x

#********************************************             Inception               ***********************************************     



class myInceptionConv(torchvision.models.Inception3):
    def __init__(self, fixed_extractor = True):
        super().__init__(torchvision.models.inception_v3(pretrained=True))
        self.load_state_dict(torch.utils.model_zoo.load_url(
            'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)'))
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False


class MyInceptiontDens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=64, out_features=32)
        self.dens2 = torch.nn.Linear(in_features=32, out_features=nb_out)
    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = self.dens2(x)
        return x



class MyInception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = myInceptionConv()
        self.mrnd = MyInceptiontDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x 
        
#********************************************           Predicting Features       *********************************************** 

def ExtractFeatures(dset_loaders, model,use_gpu=False,):
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


def getPrediction(result_valid):
    _, predicted = torch.max(result_valid, 1)
    return predicted 

def tensor2numpy(result):
    if result.type() == 'torch.cuda.LongTensor':
        result = result.cpu()
    return result.numpy()
    