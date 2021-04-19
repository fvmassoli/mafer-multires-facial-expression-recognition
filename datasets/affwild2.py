import os
import PIL
import cv2
import h5py
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os import listdir
from os.path import join, exists
from prettytable import PrettyTable, from_csv

import torch
import torchvision

from base import BaseDataset


class AffWild2Dataset(BaseDataset):
    """Class to handle AffWild2 data.
    The Affwild2 dataset is used to train/valid the model since there is not any test set available yet
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, mode: bool, multi_res: bool = False, valid_resolution: int = -1, test: bool = False, transforms: torchvision.transforms = None) -> None:
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        multi_res : bool
            If True, the model is trained with multi-resolution images
        valid_resolution : int
            Resolution at which downsample images while testing the model
        test : bool
            True if the model is under test
        transforms : torchvision.transforms
            Data augmentation transforms

        """
        super(AffWild2Dataset, self).__init__()

        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.transforms = transforms
        self.multi_res = multi_res
        self.valid_resolution = valid_resolution
        self.test = test
        
        self.mode = mode
        self.db_mode = 'train'
        
        self.idx_to_expression = {
                                0: 'Neutral',
                                1: 'Anger',
                                2: 'Disgust',
                                3: 'Fear',
                                4: 'Happiness',
                                5: 'Sadness',
                                6: 'Surprise',
                                -1: 'None',
                            }

        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'affwild2_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Database not found... creating it')
            self.__create_database(db_fname=db_fname)
        
        self.db = pd.read_csv(db_fname)
        
        # Remove all the entries which have expression label equals to -1
        self.db = self.db[self.db['expression'] != -1]

        # Print data statistics
        self.__dataset_statistics()
        
        self.set_usage()
        
    def __create_database(self, db_fname: str):
        """Create the database and save it in a .csv file.
        Following the instruction by the affwild2 team, the name of each frame corresponds
        to the position of the label in the corresponding file, i.e., the frame named video_id/00123.jpg
        correspond to the label in the position 123 of the annotation file video_id.txt.
        At the position 0 of each annotation file there is a string reporting the name of the expressions,
        that is why the frame always start from the number 00001.jpg
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        """
        train_frames_path = []
        train_video_id = []
        train_expr = []

        valid_frames_path = []
        valid_video_id = []
        valid_expr = []

        frames_base_path = join(self.dataset_folder, 'cropped_images_aligned/cropped_aligned/')

        for set_split in ['Training_Set', 'Validation_Set']:
            annotations_base_dir = join(self.dataset_folder, 'annotations/EXPR_Set', set_split)
            annotations_files = listdir(annotations_base_dir)

            for ff in tqdm(annotations_files, total=len(annotations_files), desc=f"Working on {set_split} set", leave=False):
                # read the annotation for each frame in the ff video
                annot = open(join(annotations_base_dir, ff), 'r')
                lines = np.asarray(list(map(lambda x: int(x.rstrip()) if 'Neutral' not in x else x.rstrip(), annot.readlines())))
                annot.close()

                # Convert the frame name into integers so to use them as indices to get the proper expression
                frames = list(map(lambda x: x.split('.')[0], filter(lambda x: '.jpg' in x, listdir(join(frames_base_path, ff.split('.')[0])))))
                frames_int = list(map(lambda x: int(x.split('.')[0]), filter(lambda x: '.jpg' in x, listdir(join(frames_base_path, ff.split('.')[0])))))
                
                if set_split == 'Training_Set':
                    train_video_id.extend([ff.split('.')[0] for _ in frames])
                    train_frames_path.extend([join(frames_base_path, ff.split('.')[0], frame+'.jpg') for frame in frames])
                    # use the indices into the 'frames' list to get the annotations for each face
                    train_expr.extend(lines[frames_int])
                else:
                    valid_video_id.extend([ff.split('.')[0] for _ in frames])
                    valid_frames_path.extend([join(frames_base_path, ff.split('.')[0], frame+'.jpg') for frame in frames])
                    # use the indices into the 'frames' list to get the annotations for each face
                    valid_expr.extend(lines[frames_int])

        train_expr = np.asarray(train_expr)
        train_video_id = np.asarray(train_video_id)
        valid_expr = np.asarray(valid_expr)
        valid_video_id = np.asarray(valid_video_id)

        # Create the dataframe and save it into a .csv file
        df = pd.DataFrame(
                    data=dict(
                            path=np.hstack([train_frames_path, valid_frames_path]),
                            video_id=np.hstack([train_video_id, valid_video_id]),
                            expression=np.hstack([train_expr, valid_expr]),
                            train=np.hstack([np.ones_like(train_expr, dtype=np.int), np.zeros_like(valid_expr, dtype=np.int)])
                        ),
                    index=None
                )
        df.to_csv(db_fname)
        self.logger.info(f'Database saved at: {db_fname}')

    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'\nAvailable expression: {self.idx_to_expression.values()}\n')
        
        info_table = PrettyTable()
        info_table.title = 'Multi-resolution training info'
        info_table.field_names = ['Multi-resolution', 'Validation Resolution']
        info_table.add_row([self.multi_res, self.valid_resolution])
        print(info_table)

        # Print subsample from dataframe to show its structure
        stats_table = PrettyTable()
        stats_table.title = 'Database structure'
        stats_table.field_names = self.db.columns[1:] # We don't need the index
        for idx, row in self.db.iterrows():
            stats_table.add_row([row.path, row.video_id, row.expression, row.train])
            if idx == 3: break # idx starts at 1
        if not self.test: print(stats_table)
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.title = 'Sample for training and validation'
        stats_table.field_names = ['Train samples', 'Valid samples', 'Number of expressions']
        stats_table.add_row([len(self.db[self.db['train'] == 1]), len(self.db[self.db['train'] == 0]), len(self.db.expression.unique())])
        print(stats_table)
        
        def eval_class_cardinality(expressions: list, train: bool) -> [list, list]:
            cardinality = [len(self.db[(self.db['expression']==expr) & (self.db['train']==int(train))]) for expr in expressions]
            frac = [len(self.db[(self.db['expression']==expr) & (self.db['train']==int(train))])/len(self.db[self.db['train']==int(train)]) for expr in expressions]
            return cardinality, frac

        expressions = self.db.expression.unique()
        expressions.sort()
        
        # Print expression statistics --> Training split
        stats_table = PrettyTable()
        stats_table.title = 'Samples for each expression (Training)'
        stats_table.field_names = [self.idx_to_expression[int(label)] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=True)
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)
        
        # Print expression statistics --> Validation split
        stats_table2 = PrettyTable()
        stats_table2.title = 'Samples for each expression (Validation)'
        stats_table2.field_names = [self.idx_to_expression[int(label)] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=False)
        stats_table2.add_row(cardinality)
        stats_table2.add_row(frac)

        if not self.test: 
            print(stats_table)
            print(stats_table2)
            
    def __lower_resolution(self, img: PIL.Image) -> PIL.Image:
        """Resize image to a random resolution.
        
        Parameters
        ----------
        img : PIL.Image
            Image to be resized

        Returns
        ------
        img : PIL.Image
            Resized image
        
        """
        w_i, h_i = img.size
        r = h_i/float(w_i)
        if self.train:
            res = torch.rand(1).item()
            res = 3 + 5*res
            res = 2**int(res)
        else:
            res = self.valid_resolution
        if res >= w_i or res >= h_i:
            return img
        if h_i < w_i:
            h_n = res
            w_n = h_n/float(r)
        else:
            w_n = res
            h_n = w_n*float(r)
        img2 = img.resize((int(w_n), int(h_n)), Image.BILINEAR)
        img2 = img2.resize((w_i, h_i), Image.BILINEAR)
        return img2

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression
        train : int
            Integer that indicates if dataset is in train mode (1) or not (0)

        """
        path = self.db.path.iloc[idx]
        expression = self.db.expression.iloc[idx]
        train = self.db.train.iloc[idx]

        # Load the image as a PIL.Image object
        img = self.loader(path)
        
        # Downsample the image if multi-resolution training
        if self.multi_res:
            img = self.__lower_resolution(img=img)
        
        # Apply data augmentation transformation
        img = img if self.transforms is None else self.transforms(img)
        
        return img, expression, train