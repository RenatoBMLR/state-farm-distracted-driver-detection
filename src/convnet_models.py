import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F

nb_out = 10;


#********************************************             ResNet                 ************************************************ 

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
        self.dens1 = torch.nn.Linear(in_features=6400, out_features=512)
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

# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

# http://andersonjo.github.io/artificial-intelligence/2017/05/13/Transfer-Learning/


class MyInceptionConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyInceptionConv,self).__init__()
        
        self.model = torchvision.models.inception_v3(pretrained=True)
        self.model.aux_logits= False # AH CARAI !!!! 


        n_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(n_features, 2500)   
        
        if fixed_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        
    def forward(self, x):
        x = self.model(x)
        return x



class MyInceptiontDens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=2500, out_features=512)
        self.dens2 = torch.nn.Linear(in_features=512, out_features=128)
        self.dens3 = torch.nn.Linear(in_features=128, out_features=nb_out)
        
    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = self.dens2(x)
        x = torch.nn.functional.selu(x)
        x = self.dens3(x)

        return x



class MyInception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyInceptionConv()
        self.mrnd = MyInceptiontDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x 



#********************************************             DenseNet               ***********************************************     
# http://pytorch.org/docs/0.2.0/_modules/torchvision/models/densenet.html

# https://discuss.pytorch.org/t/densenet-transfer-learning/7776


class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        original_model = torchvision.models.densenet161(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x
        
class MyDenseNetDens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=2208, out_features=512)
        self.dens2 = torch.nn.Linear(in_features=512, out_features=128)
        self.dens3 = torch.nn.Linear(in_features=128, out_features=nb_out)
        
    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = self.dens2(x)
        x = torch.nn.functional.selu(x)
        x = self.dens3(x)

        return x



class MyDenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyDenseNetConv()
        self.mrnd = MyDenseNetDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x 
    
    
