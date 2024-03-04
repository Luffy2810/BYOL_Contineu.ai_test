import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.Resnet import ResNet18,MLPHead
import seaborn as sns
import torch
import torch.nn as nn
import os
from tqdm import tqdm
torch.cuda.set_device(1)
from dataset.stl10_dataloader import get_linear_dataloader,get_test_dataloader
class Vis():
    def __init__(self,model_name):
        self.tsne = TSNE()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader_training_dataset=get_linear_dataloader()
        self.dataloader_testing_dataset=get_test_dataloader()
        
        self.resnet=ResNet18().to(self.device)
        self.predictor = MLPHead(in_channels=self.resnet.projection.net[-1].out_features,mlp_hidden_size=512, projection_size=128).to(self.device)
        self.model_name=model_name
        self.folder_name="../results/stl/"+model_name+"/best_cp/"
        self.load(self.model_name)
        self.make_folder()
    def load(self,model_name):
        self.resnet.load_state_dict(torch.load(self.folder_name+"modelq.pth"))
        # self.predictor.load_state_dict(torch.load(self.folder_name+"predictor.pth"))
        print("Model Loaded!")

    def make_folder(self):
            self.save_folder='../results/'+'stl/'+self.model_name+'/vis/'
            os.makedirs(self.save_folder,exist_ok=True)

    def plot_vecs_n_labels(self,v,labels,fname):
        fig = plt.figure(figsize = (10, 10))
        plt.axis('off')
        sns.set_style("darkgrid")
        sns.scatterplot(x=v[:,0],y= v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 10))
        
        plt.savefig(fname)
        plt.close()



    def tsnevis(self):
        self.resnet.eval()
        self.predictor.eval()
        
        for (_, sample_batched) in enumerate(tqdm(self.dataloader_training_dataset)):
            x = sample_batched['image']
            x = x.to(self.device)
            y = self.resnet(x)
            # y=self.predictor(y)
            
            y_tsne = self.tsne.fit_transform(y.cpu().data)
            labels = sample_batched['label']
            self.plot_vecs_n_labels(y_tsne,labels,self.save_folder+'/tsne_train_last_layer.png')


        
        for (_, sample_batched) in enumerate(tqdm(self.dataloader_testing_dataset)):
            x = sample_batched['image']
            x = x.to(self.device)
            y = self.resnet(x)
            # y=self.predictor(y)
            y_tsne = self.tsne.fit_transform(y.cpu().data)
            labels = sample_batched['label']
            self.plot_vecs_n_labels(y_tsne,labels,self.save_folder+'/tsne_test_last_layer.png')



        self.predictor.net = nn.Sequential(*list(self.predictor.net.children())[:-3])


        for (_, sample_batched) in enumerate(tqdm(self.dataloader_training_dataset)):
            x = sample_batched['image']
            x = x.to(self.device)
            y = self.resnet(x)
            # y=self.predictor(y_intermediate)
            y_tsne = self.tsne.fit_transform(y.cpu().data)
            labels = sample_batched['label']
            self.plot_vecs_n_labels(y_tsne,labels,self.save_folder+'tsne_train_hidden_last_layer.png')


        for (_, sample_batched) in enumerate(tqdm(self.dataloader_testing_dataset)):
            x = sample_batched['image']
            x = x.to(self.device)
            y = self.resnet(x)
            # y=self.predictor(y)
            y_tsne = self.tsne.fit_transform(y.cpu().data)
            labels = sample_batched['label']
            self.plot_vecs_n_labels(y_tsne,labels,self.save_folder+'tsne_test_hidden_last_layer.png')


if __name__=='__main__':
    vis=Vis('2024_02_05-09_40_15_PM')
    vis.tsnevis()

    