import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os.path import join, exists

from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
    

def main(args):
    device = init_seeds(args.seed)

     ## Init the datasets
    _, _, feat_dataset = init_datasets(
                                    dataset_name=args.dataset_name, 
                                    dataset_folder=args.dataset_folder, 
                                    exp_folder=args.dataset_folder 
                                )
    
    if args.dataset_name != 'rafdb':
        mean = np.load(join(args.dataset_folder, 'mean.npy'))
    else:
        mean = np.asarray([102.15835, 114.51117, 146.58075])
    
    feat_dataset.set_transforms(get_transforms(train=False, dataset_name=args.dataset_name, data_mean=mean))
    loader = init_loader(dset=feat_dataset, batch_size=args.batch_size, shuffle=False)

    ## Load the model 
    model = get_model(
                    model_base_path=args.model_base_path, 
                    num_classes=args.num_classes, 
                    model_checkpoint=args.model_checkpoint,
                    training_config='finetune', # irrelevant here
                    load_last_layer=True
                )
    model.eval().to(device)

    df = pd.DataFrame()

    nb_images = len( feat_dataset.db)

    out_folder = f'./features_extraction/{args.dataset_name}'
    if not exists(out_folder):
        os.makedirs(out_folder)

    m_ckp_shorten = args.model_checkpoint.split('/')[6]

    correct = 0
    n_processed = 0
    with torch.no_grad():
        with h5py.File(join(out_folder, f'features_{args.dataset_name}_test_set_{m_ckp_shorten}.hdf5'), 'w') as ff:
            for b_idx, (x,y) in enumerate(tqdm(loader, total=len(loader), desc='Feature extraction', leave=False), 1):
                
                x, y = x.to(device), y.to(device)
                
                features, predictions = model(x)

                batch_size, feature_dims = features.shape
                dset = ff.require_dataset(f'features_{args.dataset_name}', (nb_images, feature_dims), dtype='float32', chunks=(50, feature_dims))
                dset[n_processed:n_processed + batch_size, :] = features.detach().cpu().numpy()
                n_processed += batch_size

                correct += predictions.max(-1)[1].eq(y).sum().item()

                for idx in range(batch_size):
                    df = df.append({
                                'gt_label': y[idx].int().item(),
                                'prediction': predictions.max(-1)[1][idx].int().item(),
                                'correct': predictions.max(-1)[1][idx].item()==y[idx].item()
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
    parser.add_argument('-dn', '--dataset-name', choices=('fer2013', 'rafdb'), default='fer2013', help='Dataset\'s name (default: fer2013)')
    parser.add_argument('-of', '--out-dataset-folder', help='Folder where to save dataset csv and mean(in case of no permission)')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size (default: 4)')

    args = parser.parse_args()
    main(args)
