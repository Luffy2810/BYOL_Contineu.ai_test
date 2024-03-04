import torch
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
from PIL import Image
import random
import glob
from PIL import ImageFilter
import cv2
import numpy as np
class GaussianBlur:


    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SSLDataset(Dataset):
    def __init__(self, split=None):
        train_names = glob.glob('../../../data/output/train/*/*.jpeg',recursive=True)
        train_names.extend(glob.glob('../../../data/output/train/*/*.jpg',recursive=True))
        self.train_names=sorted(train_names)
        self.train_labels = [x.split('/')[6] for x in self.train_names]
        self.train_tokens=self.convert_labels_to_tokens(self.train_labels)

        self.val_names= [name for name in train_names if not name.startswith('../../../data/output/train/unlabelled/')]
        self.val_labels = [x.split('/')[6] for x in self.val_names]
        self.val_tokens=self.convert_labels_to_tokens(self.val_labels)
        print (self.val_tokens)
        test_names = glob.glob('../../../data/output/val/*/*.jpeg',recursive=True)
        test_names.extend(glob.glob('../../../data/output/val/*/*.jpg',recursive=True))
        self.test_names=sorted(test_names)
        self.test_labels = [x.split('/')[6] for x in self.test_names]
        self.test_tokens=self.convert_labels_to_tokens(self.test_labels)
        self.split=split
        if self.split=="val":
            print (self.val_tokens)
        if self.split=="test":
            print (self.test_tokens)
    def convert_labels_to_tokens(self,labels):
        list_set = set(labels)
        tokens = sorted(list(list_set))
        word_to_idx = {word: i for i, word in enumerate(tokens)}
        return word_to_idx


    def __len__(self):
        if self.split==None:
            return len(self.train_names)
        elif  self.split=="linear":
            return len(self.val_names)    
        elif self.split=="val" or self.split=="test":
            return len(self.test_names)    

    def pil_loader(self,path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_color_distortion(self,s=1.0):
        # color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
        # rnd_gray = T.RandomGrayscale(p=0.2)
        equalize=T.RandomEqualize(p=1)
        affine=T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10)
        rnd_affine=T.RandomApply([affine],p=0.8)
        blur=T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        flip=T.RandomHorizontalFlip()
        color_distort = T.Compose([equalize,rnd_affine,T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),flip])

        return color_distort

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((46.18/255, 46.18/255, 46.18/255), (52.42/255, 52.42/255, 52.42/255))(res)
        return res

    def mutate_image(self, img):
        res = T.RandomResizedCrop([int(512),int(512)])(img)
        res = self.get_color_distortion(1)(res)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split==None:
            img_name =  (self.train_names[idx])
            image = self.pil_loader(img_name)
            label = self.train_labels[idx]
            label=self.train_tokens[label]


            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}


        elif self.split=="val":
            img_name =  (self.test_names[idx])
            image = self.pil_loader(img_name)
            label = self.test_labels[idx]
            label=self.test_tokens[label]

            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}

        elif self.split=="linear":
            img_name =  (self.val_names[idx])
            image = self.pil_loader(img_name)
            label = self.val_labels[idx]
            label=self.val_tokens[label]

            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            sample = {'image': image1, 'label': label}

        elif self.split=="test":
            img_name =  (self.test_names[idx])
            image = self.pil_loader(img_name)
            label = self.test_labels[idx]
            label=self.test_tokens[label]

            image1 = res = T.Resize([int(512),int(512)])(image)
            equalize=T.RandomEqualize(p=1)
            image1=equalize(image1)
            image1 = self.tensorify(image1)
            sample = {'image': image1, 'label': label}


        return sample

if __name__=="__main__":
    train_names = glob.glob('../../../../data/output/train/*/*.jpeg',recursive=True)
    print (train_names)
    # ds=SSLDataset(split="linear")
    # dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=20)
    # for samples in dl:
    #     # print ()
    #     # img=np.mean(samples['image1'][5].numpy())
    #     print (samples['label'])
    #     print (img)
    #     # cv2.imshow("a",samples['image1'][0].numpy())
    #     # cv2.waitKey(0)
    #     break