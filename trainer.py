import h5py
import logging
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from tensorboardX import SummaryWriter


import torchvision
from torchvision.transforms import Compose, GaussianBlur, RandomErasing, RandomRotation, Resize, RandomHorizontalFlip, RandomGrayscale, RandomVerticalFlip
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Lambda, RandomPerspective, ColorJitter, FiveCrop, TenCrop, ToPILImage

from utils import eval_metrics, eval_score_for_competition


def train(model: torch.nn.Module, loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim, training_classes_weights: np.array, epochs: int, train_iters: int, batch_accumulation: int, patience: int, tb_writer: SummaryWriter, log_freq: int, output_folder_path: str, device: str) -> None:
    """Train the model for a given number of epochs. 
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    loader : DataLoader
        Train data loader
    valid_loader : DataLoader
        Data loader for validation
    optimizer : torch.optim
        Optimizer
    training_classes_weights : np.array
        Numpy array containing the weigths for each class in the dataset
    epochs : int
        Number of training epochs
    train_iters : int
        Number of training iterations before to run a validation step
    batch_accumulation : int
        NUmber of accumulating batch iterations
    patience : int
        Validation step before to drop the learning rate if metric reached a plateau
    tb_writer : SummaryWriter
        Writer on tensorboard
    log_freq : int
        Number of training iterations after which print training stats
    output_folder_path : str
        Path to output folder to save model checkpoints
    device : str
        Device on which to run the computations
    
    """
    logger = logging.getLogger()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience, threshold=0.001, verbose=1)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(training_classes_weights).float())
    criterion.to(device)

    model.to(device).train()
    
    it_t = 0
    it_v = 0
    i = 0

    # Validation statistics
    best_val_acc = 0.
    
    for epoch in range(epochs):
        # Training statistics
        tr_loss = 0.
        correct = 0
        n = 0

        model.zero_grad()
        optimizer.zero_grad()

        ba = 1

        pbar = tqdm(loader, total=len(loader), desc=f'Training at epoch: {epoch}', leave=False)
        for idx, (x, y) in enumerate(pbar, 1):
            n += x.shape[0]
        
            x, y = x.to(device), y.to(device)

            _, predictions = model(x)
            
            loss_ = criterion(predictions, y)       
            loss_.backward()
            
            tr_loss += loss_.item()
            correct += predictions.max(-1)[1].eq(y).sum().item()

            if ba % batch_accumulation == 0:
                ba = 1
                optimizer.step()
                optimizer.zero_grad()
            else:
                ba += 1
            
            pbar.set_postfix(Train_loss=f'{tr_loss/idx:.4f}', Accuracy=f'{correct/n * 100:.2f}%')
            
            # Perform a validation run every train_iters trainin iterations
            if idx % (len(loader)//2) == 0:
                val_loss, val_acc = validate(model=model, loader=valid_loader, device=device)
                
                # Update the learning rate scheduler
                scheduler.step(val_acc)

                tqdm.write(
                        f'####'
                        f'\tValidation at iter ([{idx}]/[{len(loader)}])'
                        f'\tValid loss: {val_loss:.4f} --- Accuracy: {val_acc * 100:.2f}%'
                        f'\t ####'
                    )
                
                if tb_writer is not None:
                    tb_writer.add_scalar('valid/loss', val_loss, it_v)
                    tb_writer.add_scalar('valid/acc', val_acc, it_v)
                    it_v += 1

                # If the current model is the best so far, then save a checkpoint
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    
                    ckp_path = join(output_folder_path, 'best_model_ckp_%d.pt'%i)
                    i+=1

                    tqdm.write(
                            f'####'
                            f'\tSaving best model at {ckp_path}'
                            f'\t####'
                        )
                    
                    torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'train_iter': idx,
                            'best_acc': best_val_acc

                        },
                        ckp_path
                    )
                
                # Set model in train mode after validation
                model.train()

            # Print training stats and save on tensorboard
            if idx % log_freq == 0:
                tqdm.write(
                        f'Train at epoch {epoch} ([{idx}]/[{len(loader)}])'
                        f'\tTrain loss: {tr_loss/idx:.4f} --- Accuracy: {correct/n * 100:.2f}%'
                    )
                
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', tr_loss/idx, it_t)
                    tb_writer.add_scalar('train/acc', correct/n, it_t)
                    it_t += 1


def validate(model: torch.nn.Module, loader: DataLoader, device: str) -> [float, float]:
    """Validate the model. 
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    loader : DataLoader
        Data loader
    device : str
        Device on which to run the computations
    
    Returns
    ------
    model : torch.nn.Model
        Trained model

    """
    n = 0
    loss = 0.
    correct = 0

    criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, total=len(loader)//3, desc='Model validation', leave=False), 1):
            n += x.shape[0]
            
            x, y = x.to(device), y.to(device)

            _, predictions = model(x)
            
            loss_ = criterion(predictions, y)
            loss += loss_.item()

            correct += predictions.max(-1)[1].eq(y).sum().item()

            # Test only on a portion of the entire validation set
            if idx % (len(loader) // 3) == 0: break

    return loss/idx, correct/n


def train_over_epoch(fold: int, model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim, criterion: torch.nn.Module, batch_accumulation: int, tb_writer: SummaryWriter, log_freq: int, device: str, epoch: int) -> None:
    """Train the model for a given number of epochs. 
    
    Parameters
    ----------
    fold : int
        Current fold index
    model : torch.nn.Module
        The model to be trained
    loader : DataLoader
        Train data loader
    optimizer : torch.optim
        Optimizer
    criterion : torch.nn.Module
        Criterion to apply to evaluate the loss
    batch_accumulation : int
        NUmber of accumulating batch iterations
    tb_writer : SummaryWriter
        Writer on tensorboard
    log_freq : int
        Number of training iterations after which print training stats
    device : str
        Device on which to run the computations
    epoch : int
        Current epoch
    
    """
    logger = logging.getLogger()

    model.train()
    
    # Training statistics
    tr_loss = 0.
    correct = 0
    n = 0

    model.zero_grad()
    optimizer.zero_grad()

    ba = 1

    pbar = tqdm(loader, total=len(loader), desc=f'Training at epoch: {epoch}')
    for idx, (x, y) in enumerate(pbar, 1):   
        n += x.shape[0]
        
        x, y = x.to(device), y.to(device)
        
        _, predictions = model(x)

        loss_ = criterion(predictions, y) 
        loss_.backward()

        tr_loss += loss_.item()

        correct += predictions.max(-1)[1].eq(y).sum().item()
            
        if ba % batch_accumulation == 0:
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.0, norm_type=2.0)
            ba = 1
            optimizer.step()
            optimizer.zero_grad()
        else:
            ba += 1

        pbar.set_postfix(Train_loss=f'{tr_loss/idx:.4f}', Accuracy=f'{correct/n * 100:.2f}%')
            
    if tb_writer is not None:
        tb_writer.add_scalar(f'train/loss_{fold}', tr_loss/len(loader), epoch)
        tb_writer.add_scalar(f'train/acc_{fold}', correct/n, epoch)


def test(model: torch.nn.Module, loader: DataLoader, device: str) -> None:
    """Test the model. 
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    loader : DataLoader
        Test data loader
    device : str
        Device on which to run the computations
    
    Returns
    ------
    model : torch.nn.Model
        Trained model

    """
    logger = logging.getLogger()

    criterion = torch.nn.CrossEntropyLoss()

    n = 0
    loss = 0.
    correct = 0

    model.to(device).eval()
    
    labels = []
    predictions = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, total=len(loader), desc='Model test', leave=False), 1):
            n += x.shape[0]
            
            x, y = x.to(device), y.to(device)

            _, predictions_ = model(x)

            loss_ = criterion(predictions_, y)       

            loss += loss_.item()
            correct += predictions_.max(-1)[1].eq(y).sum().item()

            labels.extend(y.cpu().numpy().tolist())
            predictions.extend(predictions_.max(-1)[1].detach().cpu().numpy().tolist())

    accuracy, f1_score, _ = eval_metrics(np.asarray(labels), np.asarray(predictions))
    score = eval_score_for_competition(f1_score, accuracy)

    logger.info(
            f'\n#################################################'
            f'\n#################################################'
            f'\nTest results:'
            f'\n\tLoss: {loss/len(loader):.4f} --- F1-score: {f1_score:.4f} --- Accuracy: {correct/n * 100:.2f}%'
            f'\n\tCompetition score: {score:.4f}'
            f'\n#################################################'
            f'\n#################################################'
        )

    return accuracy, f1_score, score 
