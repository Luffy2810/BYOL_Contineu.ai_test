import sys
sys.path.append('../src')
from dataset.dataloader import get_mutated_dataloader,get_val_dataloader
from model.Resnet import ResNet18,MLPHead
from model.loss import loss_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
torch.cuda.set_device(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

class BYOL():
    def __init__(self,model_load=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader_training_dataset_mutated = get_mutated_dataloader()
        self.val_dataloader=get_val_dataloader()
        # print (len(self.dataloader_training_dataset_mutated),len(self.val_dataloader))
        self.resnetq=ResNet18().to(self.device)
        self.resnetk = copy.deepcopy(self.resnetq).to(self.device)
        self.predictor = MLPHead(in_channels=self.resnetq.projection.net[-1].out_features,mlp_hidden_size=512, projection_size=128).to(self.device)
        self.optimizer = torch.optim.SGD(list(self.resnetq.parameters()) + list(self.predictor.parameters()),1.6, weight_decay=1.5e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.dataloader_training_dataset_mutated), eta_min=0,
                                                            last_epoch=-1)
        self.losses_train = []
        self.losses_val=[]
        
        self.num_epochs = 1200
        self.momentum=0.99
        self.min_loss=9999
        
        self.model_load=model_load
        self.date_time=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_name='dset/'+self.date_time+'/'
        if self.model_load:
            self.load_folder="../results/dset/"+self.model_load+"/best_cp/" 
            self.load()
        self.make_folder()
    def make_folder(self):
            self.save_folder='../results/'+'dset/'+self.date_time+'/'
            os.makedirs(self.save_folder)
            os.makedirs(self.save_folder+'best_cp')
            os.makedirs(self.save_folder+'last_cp')

    def get_mean_of_list(self,L):
        return sum(L) / len(L)

    def load(self):
        self.resnetq.load_state_dict(torch.load(self.load_folder+"modelq.pth"))
        self.resnetk.load_state_dict(torch.load(self.load_folder+"modelk.pth"))
        self.predictor.load_state_dict(torch.load(self.load_folder+"predictor.pth"))
        self.optimizer.load_state_dict(torch.load(self.load_folder+"optimizer.pth"))
        temp = np.load(self.load_folder+"lossesfile.npz")
        self.losses_train = list(temp['arr_0'])
        print("Model Loaded!")


        
    def save(self,epoch_losses_val,epoch_losses_train,epoch):
        self.losses_val.append(self.get_mean_of_list(epoch_losses_val))
        self.losses_train.append(self.get_mean_of_list(epoch_losses_train))
        self.writer.add_scalar("imagenet/train_loss", self.get_mean_of_list(epoch_losses_train), epoch)
        self.writer.add_scalar("imagenet/val_loss", self.get_mean_of_list(epoch_losses_val), epoch)
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(self.losses_train)
        plt.legend(['Training Losses'])
        plt.savefig(self.save_folder+'last_cp/'+'train_losses.png')
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(self.losses_val)
        plt.legend(['Training Losses'])
        plt.savefig(self.save_folder+'last_cp/'+'val_losses.png')
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(self.losses_val,label="val")
        plt.plot(self.losses_train,label="train")
        plt.legend()
        plt.savefig(self.save_folder+'last_cp/'+'combined_losses.png')
        plt.close()



        if self.get_mean_of_list(epoch_losses_train)<self.min_loss:
            self.min_loss=self.get_mean_of_list(epoch_losses_train)
            torch.save(self.resnetq.state_dict(), self.save_folder+'best_cp/'+'modelq.pth')
            torch.save(self.resnetk.state_dict(), self.save_folder+'best_cp/'+'modelk.pth')
            torch.save(self.optimizer.state_dict(), self.save_folder+'best_cp/'+'optimizer.pth')
            torch.save(self.predictor.state_dict(), self.save_folder+'best_cp/'+'predictor.pth')
            np.savez(self.save_folder+'best_cp/'+"lossesfile", np.array(self.losses_train))
        torch.save(self.resnetq.state_dict(), self.save_folder+'last_cp/'+'modelq.pth')
        torch.save(self.resnetk.state_dict(), self.save_folder+'last_cp/'+'modelk.pth')
        torch.save(self.optimizer.state_dict(), self.save_folder+'last_cp/'+'optimizer.pth')
        torch.save(self.predictor.state_dict(), self.save_folder+'last_cp/'+'predictor.pth')
        np.savez(self.save_folder+'last_cp/'+"lossesfile", np.array(self.losses_train))
        if (epoch%200==0 and epoch!=0):
            os.makedirs(self.save_folder+'epoch_' +str(epoch)+ '/')
            torch.save(self.resnetq.state_dict(), self.save_folder+'epoch_' +str(epoch)+ '/'+'modelq.pth')
            torch.save(self.resnetk.state_dict(), self.save_folder+'epoch_' +str(epoch)+ '/'+'modelk.pth')
            torch.save(self.optimizer.state_dict(), self.save_folder+'epoch_' +str(epoch)+ '/'+'optimizer.pth')
            torch.save(self.predictor.state_dict(), self.save_folder+'epoch_' +str(epoch)+ '/'+'predictor.pth')

    def val(self):
        with torch.no_grad():
            epoch_losses_val=[]
            for (_, sample_batched) in enumerate(tqdm(self.val_dataloader)):
                i1 = sample_batched['image1']
                i2 = sample_batched['image2']
                i1= i1.to(self.device)
                i2 = i2.to(self.device)


                predictions_from_view_1_inter = (self.resnetq(i1))
                predictions_from_view_1=self.predictor(predictions_from_view_1_inter)
                predictions_from_view_2_inter = (self.resnetq(i2))
                predictions_from_view_2=self.predictor(predictions_from_view_2_inter)
                with torch.no_grad():
                    targets_to_view_2 = self.resnetk(i1)
                    targets_to_view_1 = self.resnetk(i2)


                loss = loss_function(predictions_from_view_1, targets_to_view_1)
                loss += loss_function(predictions_from_view_2, targets_to_view_2)

                epoch_losses_val.append(loss.mean().cpu().data.item())

        return epoch_losses_val


    def train(self):
        self.resnetq.train()
        self.predictor.train()
        self.writer = SummaryWriter()
        for epoch in range(self.num_epochs):
            print(epoch)
            epoch_losses_train = []
            for (_, sample_batched) in enumerate(tqdm(self.dataloader_training_dataset_mutated)):
                self.optimizer.zero_grad()
                i1 = sample_batched['image1']
                i2 = sample_batched['image2']
                i1= i1.to(self.device)
                i2 = i2.to(self.device)
                predictions_from_view_1 = self.predictor(self.resnetq(i1))
                predictions_from_view_2 = self.predictor(self.resnetq(i2))
                with torch.no_grad():
                    targets_to_view_2 = self.resnetk(i1)
                    targets_to_view_1 = self.resnetk(i2)
                loss = loss_function(predictions_from_view_1, targets_to_view_1)
                loss += loss_function(predictions_from_view_2, targets_to_view_2)
                epoch_losses_train.append(loss.mean().cpu().data.item())
                loss.mean().backward()
                self.optimizer.step()
                for θ_k, θ_q in zip(self.resnetk.parameters(), self.resnetq.parameters()):
                    θ_k.data.copy_(self.momentum*θ_k.data + θ_q.data*(1.0 - self.momentum))
            self.scheduler.step()
            epoch_losses_val=self.val()
            self.save(epoch_losses_val,epoch_losses_train,epoch)

if __name__=="__main__":
    byol=BYOL()
    byol.train()