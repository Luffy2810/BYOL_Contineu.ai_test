from torch.optim.lr_scheduler import StepLR
from torch import optim
import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.Resnet import ResNet18,MLPHead
import seaborn as sns
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
torch.cuda.set_device(0)
from dataset.dataloader import get_linear_dataloader,get_test_dataloader
class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(128, 16)


    def forward(self, x):
        # print (x.size())
        # print (x[0])
        x = self.fc1(x)
        return(x)



class Linear():
    def __init__(self,model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader_testing_dataset=get_test_dataloader()
        self.resnet=ResNet18().to(self.device)
        self.linear_net=LinearNet().to(self.device)
        self.model_name=model_name
        self.folder_name="../results/dset/"+model_name+"/best_cp/"
        self.load(self.model_name)
        self.make_folder()
        self.max_test_acc=0

    def load(self,model_name):
    
        self.resnet.load_state_dict(torch.load("../modelq.pth"))
        self.save_folder='../results/'+'dset/'+self.model_name+'/linear_wo_proj/'
        self.linear_net.load_state_dict(torch.load("../linear_model.pth"))
        # print (self.linear_net.state_dict())
        # self.predictor.load_state_dict(torch.load(self.folder_name+"predictor.pth"))
        print("Model Loaded!")

    def make_folder(self):
            self.save_folder='../results/'+'dset/'+self.model_name+'/linear_wo_proj/'
            os.makedirs(self.save_folder,exist_ok=True)

    def test(self):
        epoch_losses_test_linear = []
        epoch_acc_test_num_linear = 0.0
        epoch_acc_test_den_linear = 0.0

        for (_, sample_batched) in enumerate(tqdm(self.dataloader_testing_dataset)):
            
            x = sample_batched['image']
            y_actual = sample_batched['label']
            y_actual = np.asarray(y_actual)
            y_actual = torch.from_numpy(y_actual.astype('long'))
            x = x.to(self.device)
            y_actual  = y_actual.to(self.device)
            # y_intermediate = self.predictor(self.resnet(x))
            # y_intermediate = self.predictor(self.resnet(x))
            y_predicted = self.linear_net(self.resnet(x))
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
            epoch_losses_test_linear.append(loss.data.item())
            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
            epoch_acc_test_num_linear += (actual == pred).sum().item()
            epoch_acc_test_den_linear += len(actual)

        test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
        print(test_acc)



    def linear_train(self):

        self.resnet.eval()
    
        self.linear_net.eval()
        self.test()
                
if __name__=="__main__":
    linear=Linear('2024_03_03-02_21_04_AM')
    linear.linear_train()