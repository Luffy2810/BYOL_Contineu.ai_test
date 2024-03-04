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
        self.fc1 = torch.nn.Linear(512, 15)


    def forward(self, x):
        # print (x.size())
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc1(x)
        return(x)



class Linear():
    def __init__(self,model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader_training_dataset=get_linear_dataloader()
        self.dataloader_testing_dataset=get_test_dataloader()
        self.resnet=ResNet18().to(self.device)
        self.predictor = MLPHead(in_channels=self.resnet.projection.net[-1].out_features,mlp_hidden_size=512, projection_size=128).to(self.device)
        self.linear_net=LinearNet().to(self.device)
        self.model_name=model_name
        self.folder_name="../results/dset/"+model_name+"/best_cp/"
        self.load(self.model_name)
        self.make_folder()
        # self.linear_optimizer = optim.SGD(self.linear_net.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
        self.linear_optimizer = optim.LBFGS(self.linear_net.parameters(), lr=0.3)
        self.resnet_opitimzer=optim.SGD(self.resnet.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
        self.linear_scheduler = StepLR(self.linear_optimizer, step_size=7, gamma=0.1)
        self.resnet_scheduler = StepLR(self.resnet_opitimzer, step_size=7, gamma=0.1)
        self.num_epochs=28
        self.max_test_acc=0
        self.losses_train_linear = []
        self.acc_train_linear = []
        self.losses_test_linear = []
        self.acc_test_linear = []
    def load(self,model_name):
    
        self.resnet.load_state_dict(torch.load(self.folder_name+"modelq.pth"))
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # self.save_folder='../results/'+'dset/'+self.model_name+'/linear_wo_proj/'
        # self.linear_net.load_state_dict(torch.load(self.save_folder+"model.pth"))
        # self.predictor.load_state_dict(torch.load(self.folder_name+"predictor.pth"))
        print("Model Loaded!")

    def make_folder(self):
            self.save_folder='../results/'+'dset/'+self.model_name+'/linear_wo_proj/'
            os.makedirs(self.save_folder,exist_ok=True)

    def get_mean_of_list(self,L):
        return sum(L) / len(L)
    def save(self):
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train_linear)
        plt.plot(losses_test_linear)
        plt.legend(['Training Losses', 'Testing Losses'])
        plt.savefig(self.save_folder+'losses.png')
        plt.close()
        writer.add_scalar("linear_wo_proj/accuracy_train", epoch_acc_train_num_linear / epoch_acc_train_den_linear, epoch)
        writer.add_scalar("linear_wo_proj/accuracy_test", test_acc, epoch)
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(acc_train_linear)
        plt.plot(acc_test_linear)
        plt.legend(['Training Accuracy', 'Testing Accuracy'])
        plt.savefig(self.save_folder+'accuracy.png')
        plt.close()

    def train(self):
        epoch_losses_train_linear = []
        epoch_acc_train_num_linear = 0.0
        epoch_acc_train_den_linear = 0.0

        for (_, sample_batched) in enumerate(tqdm(self.dataloader_training_dataset)):
            def closure():
                self.linear_optimizer.zero_grad()
                # self.resnet_opitimzer.zero_grad()
                
                x = sample_batched['image']
                y_actual = sample_batched['label']
                # print (y_actual)
                x = x.to(self.device)
                self.y_actual  = y_actual.to(self.device)
                # y_intermediate = self.predictor(self.resnet(x))
                # y_predicted=self.linear_net(y_intermediate)
                self.y_predicted = self.linear_net(self.resnet(x))
                loss = nn.CrossEntropyLoss()(self.y_predicted, self.y_actual)
                epoch_losses_train_linear.append(loss.data.item())
                loss.backward()
                return loss
            self.linear_optimizer.step(closure)
            # self.resnet_opitimzer.step()
            pred = np.argmax(self.y_predicted.cpu().data, axis=1)
            actual = self.y_actual.cpu().data
            epoch_acc_train_num_linear += (actual == pred).sum().item()
            epoch_acc_train_den_linear += len(actual)
        self.losses_train_linear.append(self.get_mean_of_list(epoch_losses_train_linear))
        self.acc_train_linear.append(epoch_acc_train_num_linear / epoch_acc_train_den_linear)
        self.linear_scheduler.step()

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


        self.losses_test_linear.append(self.get_mean_of_list(epoch_losses_test_linear))
        self.acc_test_linear.append(epoch_acc_test_num_linear / epoch_acc_test_den_linear)
        print("Epoch completed")
        if test_acc >= self.max_test_acc:
            self.max_test_acc = test_acc
            torch.save(self.linear_net.state_dict(), self.save_folder+'model_nop.pth')
            # torch.save(self.linear_optimizer.state_dict(), self.save_folder+'optimizer.pth')

    def linear_train(self):
        for epoch in range(self.num_epochs):
            print (epoch)
            self.resnet.eval()
            # self.resnet.train()
            self.linear_net.train()
            self.predictor.train()
            self.train()

            self.resnet.eval()
            self.predictor.eval()
            self.linear_net.eval()
            self.test()
        print (self.max_test_acc)
if __name__=="__main__":
    linear=Linear('2024_03_03-02_13_37_AM')
    linear.linear_train()