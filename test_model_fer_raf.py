import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os.path import join, exists
import sklearn.metrics as skm

from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
    

def main(args):
    device = init_seeds(args.seed)

     ## Init the datasets
    _, _, test_dataset = init_datasets(
                                    dataset_name=args.dataset_name, 
                                    dataset_folder=args.dataset_folder, 
                                    exp_folder=args.dataset_folder 
                                )
    
    if args.dataset_name != 'rafdb':
        mean = np.load(join(args.dataset_folder, 'mean.npy'))
    else:
        mean = np.asarray([102.15835, 114.51117, 146.58075])

    test_dataset.set_transforms(get_transforms(train=False, dataset_name=args.dataset_name, data_mean=mean))
    loader = init_loader(dset=test_dataset, batch_size=args.batch_size, shuffle=False)

    ## Load the model 
    model = get_model(
                    model_base_path=args.model_base_path, 
                    num_classes=args.num_classes, 
                    model_checkpoint=args.model_checkpoint,
                    training_config='finetune', # irrelevant here
                    load_last_layer=True
                )
    model.eval().to(device)

    correct = 0
    nb_images = 0
    labels = []
    predictions_ = []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(tqdm(loader, total=len(loader), desc='Feature extraction', leave=False), 1):
            nb_images += x.shape[0]
            
            x, y = x.to(device), y.to(device)
            
            _, predictions = model(x)

            correct += predictions.max(-1)[1].eq(y).sum().item()

            labels.extend(y.cpu().numpy())
            predictions_.extend(predictions.max(-1)[1].cpu().numpy())

    sklearn_accuracy = skm.accuracy_score(labels, predictions_)
    
    print(f'Model accuracy on the test set (manual):  {(correct/nb_images)*100.:.2f}%')
    print(f'Model accuracy on the test set (sklearn): {sklearn_accuracy*100.:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CRFER')

    ## General
    parser.add_argument('-s', '--seed', type=int, default=17)

    ## Model selection
    parser.add_argument('-ck', '--model_checkpoint', help='Path to model checkpoint')
    parser.add_argument('-bp', '--model_base_path', help='Path to base model checkpoint')
    parser.add_argument('-nc', '--num_classes', type=int, default=7, help='Number of training classes (default: 7)')

    ## Extraction
    parser.add_argument('-df', '--dataset-folder', help='Path to main data folder')
    parser.add_argument('-dn', '--dataset-name', choices=('fer2013', 'rafdb'), default='fer2013', help='Dataset\'s name (default: fer2013)')
    parser.add_argument('-of', '--out-dataset-folder', help='Folder where to save dataset csv and mean(in case of no permission)')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size (default: 4)')

    args = parser.parse_args()
    main(args)
