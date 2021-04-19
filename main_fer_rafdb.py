import os
import sys
import h5py
import random
import logging
import argparse
import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.model_selection import KFold

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from tensorboardX import SummaryWriter

from utils import *
from trainer import train, train_over_epoch, validate, test


def main(args):
    device = init_seeds(args.seed)

    ## Create output folders
    output_main_folder, exp_folder = create_experiment_folder(args=args, dataset_name=args.dataset_name)
    
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s | %(message)s",
                    handlers=[
                        logging.FileHandler(os.path.join(exp_folder, f'training_{args.dataset_name}.log')),
                        logging.StreamHandler()
                    ])
    
    ## Init tensorboard writer
    tb_writer = SummaryWriter(join(output_main_folder, 'tb_run', exp_folder.split('/')[-1]))

    mean = np.load(join(args.dataset_folder, 'mean.npy')) if args.training_config == 'finetune' else np.asarray([91.4953, 103.8827, 131.0912])
    print(f'Channels mean: {mean}')

    ## Init the datasets
    train_dataset, valid_dataset, test_dataset = init_datasets(
                                                            dataset_name=args.dataset_name, 
                                                            dataset_folder=args.dataset_folder, 
                                                            exp_folder=args.dataset_folder # Save the .csv in the same folder where data are
                                                        )

    train_dataset.set_transforms(get_transforms(train=True, dataset_name=args.dataset_name, data_mean=mean))
    valid_dataset.set_transforms(get_transforms(train=False, dataset_name=args.dataset_name, data_mean=mean))
    test_dataset.set_transforms(get_transforms(train=False, dataset_name=args.dataset_name, data_mean=mean))

    ## Load the model 
    model = get_model(
                    model_base_path=args.model_base_path, 
                    num_classes=args.num_classes, 
                    model_checkpoint=args.model_checkpoint,
                    training_config=args.training_config,
                    load_last_layer=not args.train # In case we only want to test the model, we need to load the last layer too
                )
    model.to(device)

    if args.train:
        ## Get classes weigths to weight the training loss to take care of the classes unbalance 
        training_classes_weights = train_dataset.get_training_classes_weights()
        ## Feed the criterion with the training weights
        criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(training_classes_weights).float())
        ## Move to criterion, the weights, on the device
        criterion.to(device)

        ## Init the optimizer
        optimizer = init_optimizer(
                                model=model, # Only train the last layer
                                optimizer_name=args.optimizer, 
                                lr=args.lr, 
                                lr_ft=args.lr_ft,
                                momentum=args.momentum, 
                                nesterov=args.nesterov, 
                                weight_decay=args.weight_decay,
                                amsgrad=args.amsgrad, 
                                beta1=args.beta1, 
                                beta2=args.beta2,
                                training_config=args.training_config
                            )

        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, threshold=0.001, verbose=1)

        # best_acc = 0
        
        train_loader = init_loader(dset=train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = init_loader(dset=valid_dataset, batch_size=args.batch_size)

        ckp_path = join(exp_folder, 'best_model_ckp.pt')
        
        train(
            model=model, 
            loader=train_loader, 
            valid_loader=valid_loader, 
            optimizer=optimizer, 
            training_classes_weights=training_classes_weights,
            epochs=args.epochs, 
            train_iters=-1, 
            batch_accumulation=args.batch_accumulation, 
            patience=args.patience,
            tb_writer=tb_writer, 
            log_freq=args.log_freq, 
            output_folder_path=exp_folder, 
            device=device
        )

    if args.test:
        model_checkpoint = args.model_checkpoint
        if args.train: # In case we also trained the model, we can't use args.model_checkpoint
            model_checkpoint = join(exp_folder, 'best_model_ckp.pt')

        accuracy, f1_score, challenge_score = test(
                                                model=model, 
                                                loader=init_loader(dset=test_dataset, batch_size=args.batch_size, shuffle=False), 
                                                device=device
                                            )

        if args.dataset_name == 'rafdb':
                class_accuracy(
                                model=model,
                                loader=init_loader(dset=test_dataset, batch_size=args.batch_size, shuffle=False),
                                device=device
                                )

        df = pd.DataFrame(
                        data=dict(
                                model_checkpoint=[model_checkpoint],
                                accuracy=[accuracy],
                                f1_score=[f1_score],
                                challenge_score=[challenge_score]
                            ),
                        index=None
                    )

        test_results_dir = './test_results'

        if not exists(test_results_dir):
            os.makedirs(test_results_dir)

        output_csv_path = join(test_results_dir, f'./test_results_{args.dataset_name}.csv')
        df.to_csv(output_csv_path, mode='a', header=not exists(output_csv_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CRFER')

    ## General
    parser.add_argument('-s', '--seed', type=int, default=17)
    
    ## Optimizer
    parser.add_argument('-o', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Type of optimizer (default: adam)')
    parser.add_argument('-lr', '--lr', type=float, default=1.e-2, help='Learning rate (default: 1.e-2)')
    parser.add_argument('-lrf', '--lr-ft', type=float, default=1.e-4, help='Learning rate for first layers (default: 1.e-4)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1.e-4, help='Weight decay (default: 1.e-4)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum of the SGD optimizer (default: 0.9)')
    parser.add_argument('-nt', '--nesterov', action='store_true', help='Use Nesterov with the SGD optimizer (default: False)')
    parser.add_argument('-pt', '--patience', type=int, default=10, help='Patience for the optimizer scheduler (default: 10)')
    parser.add_argument('-ams', '--amsgrad', action='store_true', help='Use AmsGrad with the Adam Optimizer (default: False)')
    parser.add_argument('-b1', '--beta1', type=float, default=0.9, help='Beat 1 for Adam (default: 0.9)')
    parser.add_argument('-b2', '--beta2', type=float, default=0.999, help='Beat 1 for Adam (default: 0.999)')
    
    ## Model selection
    parser.add_argument('-ck', '--model_checkpoint', help='Path to model checkpoint')
    parser.add_argument('-bp', '--model_base_path', help='Path to base model checkpoint')
    parser.add_argument('-nc', '--num_classes', type=int, default=7, help='Number of training classes (default: 7)')
    
    ## Training
    parser.add_argument('-df', '--dataset-folder', help='Path to main data folder')
    parser.add_argument('-dn', '--dataset-name', choices=('fer2013', 'rafdb'), default='fer2013', help='Dataset\'s name (default: fer2013)')
    parser.add_argument('-tc', '--training-config', choices=('finetune', 'trl'), default='finetune', help='Training configuration (default: finetune)')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-ba', '--batch_accumulation', type=int, default=4, help='Number of batch accumulation iterations (default: 4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('-lf', '--log_freq', type=int, default=100, help='Log frequency (default: 100)')
    parser.add_argument('-op', '--output_folder_path', default='./training_output', help='Path to output folder (default: ./training_output)')
    parser.add_argument('-tr', '--train', action='store_true', help='Train the model (default: False)')
    parser.add_argument('-tt', '--test', action='store_true', help='Test the model (default: False)') 
    
    args = parser.parse_args()

    main(args)