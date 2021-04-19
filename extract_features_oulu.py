import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os.path import join, exists

import sklearn.metrics as skm
from sklearn.model_selection import KFold

from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
    

def main(args):
    device = init_seeds(args.seed)


    mean = np.load(join(args.dataset_folder, 'mean.npy'))
    print(f'Channels mean: {mean}')

    ## Init the datasets
    train_dataset, _, test_dataset = init_datasets(
                                                dataset_name=args.dataset_name, 
                                                dataset_folder=args.dataset_folder, 
                                                exp_folder=args.dataset_folder # Save the .csv in the same folder where data are
                                            )

    test_dataset.set_transforms(get_transforms(train=False, dataset_name=args.dataset_name, data_mean=mean))

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    model_checkpoints = ['_'.join(ff.split('.')[0].split('_')[:-1])+'_'+str(ii)+'.pt' for ii, ff in enumerate(os.listdir(args.model_checkpoint), 1)]
    
    out_folder = f'./features_extraction/{args.dataset_name}'
    if not exists(out_folder):
        os.makedirs(out_folder)

    m_ckp_shorten = args.model_checkpoint.split('/')[6]
    
    df = pd.DataFrame()

    for idx, (_, test_fold_idxs) in enumerate(kf.split((np.arange(len(train_dataset)))), 1):
        ## Load the model for the current fold
        model = get_model(
                        model_base_path=args.model_base_path, 
                        num_classes=args.num_classes, 
                        model_checkpoint=os.path.join(args.model_checkpoint, model_checkpoints[idx-1]),
                        training_config='finetune', # irrelevant
                        load_last_layer=True
                    )
        model.eval().to(device)
        
        test_loader = DataLoader(
                            dataset=test_dataset,
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(test_fold_idxs),
                            num_workers=8,
                            pin_memory=device=='cuda'
                        )
                
        nb_images = len(test_fold_idxs)

        correct = 0
        n_processed = 0
        with torch.no_grad():
            with h5py.File(join(out_folder, f'features_{args.dataset_name}_test_set_{m_ckp_shorten}.hdf5'), 'a') as ff:
                for b_idx, (x,y) in enumerate(tqdm(test_loader, total=len(test_loader), desc=f'Feature extraction at fold: {idx}/{n_splits}', leave=False), 1):
                    
                    x, y = x.to(device), y.to(device)
                    
                    features, predictions = model(x)

                    batch_size, feature_dims = features.shape
                    dset = ff.require_dataset(f'features_{args.dataset_name}_fold_{idx}', (nb_images, feature_dims), dtype='float32', chunks=(50, feature_dims))
                    dset[n_processed:n_processed + batch_size, :] = features.detach().cpu().numpy()
                    n_processed += batch_size

                    correct += predictions.max(-1)[1].eq(y).sum().item()

                    for idx_ in range(batch_size):
                        df = df.append({
                                    'fold': idx,
                                    'gt_label': y[idx_].int().item(),
                                    'prediction': predictions.max(-1)[1][idx_].int().item(),
                                    'correct': predictions.max(-1)[1][idx_].item()==y[idx_].item()
                                }, 
                                ignore_index=True
                            )

                    df.to_csv(join(out_folder, f'features_{args.dataset_name}_test_set_{m_ckp_shorten}.csv'))
        
        print(f'Model accuracy on the test set: {(correct/nb_images)*100.:.2f}%')


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
    parser.add_argument('-dn', '--dataset-name', default='oulucasia', help='Dataset\'s name (default: oulucasia)')
    parser.add_argument('-of', '--out-dataset-folder', help='Folder where to save dataset csv and mean(in case of no permission)')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size (default: 4)')

    args = parser.parse_args()
    main(args)
