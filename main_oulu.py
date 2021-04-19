import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.model_selection import KFold

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

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

    test_results_dir = './test_results'
    if not exists(test_results_dir):
        os.makedirs(test_results_dir)
    output_csv_path = join(test_results_dir, f'./test_results_{args.dataset_name}.csv')

    mid_path = 'base_model' if args.model_checkpoint is None else args.model_checkpoint.split('/')[-2]
    ckp_path_ = join(exp_folder, mid_path)
    if not exists(ckp_path_):
        os.makedirs(ckp_path_)
        
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    for idx, (train_fold_idxs, test_fold_idxs) in enumerate(kf.split((np.arange(len(train_dataset)))), 1):
        ckp_path = join(ckp_path_, f'best_model_ckp_fold_{idx}.pt')
        
        ## Load the model for the current fold
        model = get_model(
                        model_base_path=args.model_base_path, 
                        num_classes=args.num_classes, 
                        model_checkpoint=args.model_checkpoint,
                        training_config=args.training_config,
                        load_last_layer=not args.train # In case we only want to test the model, we need to load the last layer too
                    )
        model.to(device)
        
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
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, threshold=0.001, verbose=1)
        
        # Sample elements randomly from a given list of ids, no replacement.
        random.shuffle(train_fold_idxs)
        train_fold_idxs_ = train_fold_idxs[int(len(train_fold_idxs)*0.15):]
        valid_fold_idxs = train_fold_idxs[:int(len(train_fold_idxs)*0.15)]

        train_subsampler = SubsetRandomSampler(train_fold_idxs_)
        valid_subsampler = SubsetRandomSampler(valid_fold_idxs)
        test_subsampler  = SubsetRandomSampler(test_fold_idxs)

        ## Get classes weigths to weight the training loss to take care of the classes unbalance 
        training_classes_weights = train_dataset.get_training_classes_weights()
        
        criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(training_classes_weights).float())
        criterion.to(device)

        train_loader = DataLoader(
                            dataset=train_dataset,
                            batch_size=args.batch_size,
                            sampler=train_subsampler,
                            num_workers=8,
                            pin_memory=device=='cuda'
                        )
        valid_loader = DataLoader(
                            dataset=valid_dataset,
                            batch_size=args.batch_size,
                            sampler=valid_subsampler,
                            num_workers=8,
                            pin_memory=device=='cuda'
                        )
        test_loader = DataLoader(
                            dataset=test_dataset,
                            batch_size=args.batch_size,
                            sampler=test_subsampler,
                            num_workers=8,
                            pin_memory=device=='cuda'
                        )
        
        if args.train:
            best_acc = 0
        
            for epoch in range(args.epochs):
                train_over_epoch(
                            fold=idx+1,
                            model=model, 
                            loader=train_loader, 
                            optimizer=optimizer,
                            criterion=criterion, 
                            batch_accumulation=args.batch_accumulation, 
                            tb_writer=tb_writer, 
                            log_freq=args.log_freq, 
                            device=device,
                            epoch=epoch
                        )
                
                val_loss, val_acc = validate(model=model, loader=valid_loader, device=device)
                scheduler.step(val_acc)
                
                print(
                    f'####'
                    f'\tValidation at epoch: {epoch+1}'
                    f'\tValid loss: {val_loss:.4f} --- Accuracy: {val_acc * 100:.2f}%'
                    f'\t ####'
                )
                
                if best_acc < val_acc:
                    best_acc = val_acc

                    print(
                        f'####'
                        f'\tSaving best model at {ckp_path}'
                        f'\t####'
                    )
                    
                    torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'best_acc': best_acc

                        },
                        ckp_path
                    )
        
            # Before testing, load the best model for the current run
            print(f'################## Loading the best model for the current run ##################')
            model = get_model(
                            model_base_path=args.model_base_path, 
                            num_classes=args.num_classes, 
                            model_checkpoint=ckp_path,
                            training_config=args.training_config,
                            load_last_layer=True
                        )

        accuracy, f1_score, score = test(
                                        model=model, 
                                        loader=test_loader, 
                                        device=device
                                    )
        
        # For each folder, save the stats for the best model
        result = pd.DataFrame(
                            data=dict(
                                    model_ckp=ckp_path,
                                    accuracy=accuracy,
                                    f1_score=f1_score,
                                    challenge_score=score,
                                    fold=idx
                                ), 
                            index=[0]
                        )
        result.to_csv(output_csv_path, mode='a', header=not exists(output_csv_path))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CRFER')

    ## General
    parser.add_argument('-s', '--seed', type=int, default=17)
    
    ## Optimizer
    parser.add_argument('-o', '--optimizer', choices=('adam', 'adamw', 'sgd'), default='adam', help='Type of optimizer (default: adam)')
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
    parser.add_argument('-dn', '--dataset-name', help='Dataset\'s name')
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
