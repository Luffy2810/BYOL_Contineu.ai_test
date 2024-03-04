from re import T
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import SSLDataset
import pandas as pd
import os
import glob
import random
random.seed(0)

def convert_labels_to_tokens(labels):
    list_set = set(labels)
    tokens = (list(list_set))
    word_to_idx = {word: i for i, word in enumerate(tokens)}
    return word_to_idx


def get_mutated_dataloader():

    training_dataset_mutated = SSLDataset(split=None)
    dataloader_training_dl = DataLoader(training_dataset_mutated, batch_size=64, shuffle=True, num_workers=8)
    return dataloader_training_dl
    
def get_linear_dataloader():
    linear_dataset = SSLDataset(split="linear")
    dataloader_linear_dl = DataLoader(linear_dataset, batch_size=128, shuffle=True, num_workers=2)
    return dataloader_linear_dl

def get_val_dataloader():
    val_dataset = SSLDataset(split="val")
    dataloader_val_dl = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
    return dataloader_val_dl

def get_test_dataloader():
    testing_dataset = SSLDataset(split="test")
    dataloader_testing_dl = DataLoader(testing_dataset, batch_size=256, shuffle=True, num_workers=2)
    return dataloader_testing_dl


