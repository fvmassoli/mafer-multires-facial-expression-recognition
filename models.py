import logging
from os.path import exists, join

import torch
import torch.nn as nn


class ModelLoader(object):
    """Claass to handle models.

    """
    def __init__(self, model_base_path: str = None, num_classes: int = 7):
        """Init the ModelLoader class.

        Parameters
        ----------
        model_base_path : str
            Path to base model for SeNet50
        num_classes : int
            Number of classes for the classification task
        
        """
        self.model_base_path = model_base_path
        self.num_classes = num_classes
        self.logger = logging.getLogger()
        
        # Load the base model
        self.model = self.__load_base_model()
        
    def __get_total_number_of_training_parameters(self) -> int:
        """Eval the numnber of model parameters to be trained.

        Returns
        ------
        n_pars : int
            Number of parameters to be trained

        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __load_base_model(self) -> nn.Module:
        """Load the base model and init the classifier layer with the correct number of output classes.

        Returns
        -------
        model : nn.Module
            The base model finetuned on the VGGFace2 dataset with a new classifier layer

        """
        assert exists(self.model_base_path), f"Base model checkpoint not found at: {self.model_base_path}"
        
        # Load the base model
        model = torch.load(self.model_base_path)
        
        # Init the classifier with the correct number of classes
        model.classifier_1 = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        
        return model

    def load_model_checkpoint(self, model_checkpoint: str, load_last_layer: bool) -> None:
        """Load model checkpoint.

        Parameters
        ----------
        model_checkpoint : str
            Path to model checkpoint
        load_last_layer : bool
            Boolean used to decide wether or not load the last layer weights. 
    
        """
        assert exists(model_checkpoint), f"Model checkpoint not found at: {model_checkpoint}"

        ckp = torch.load(model_checkpoint, map_location='cpu')
        
        if load_last_layer:
            [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in self.model.named_parameters()]
        else:
            [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in self.model.named_parameters() if 'classifier_1' not in n]
        
        for n, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
                m.running_var = ckp['model_state_dict'][n + '.running_var']
                m.running_mean = ckp['model_state_dict'][n + '.running_mean']
                m.num_batches_tracked = ckp['model_state_dict'][n + '.num_batches_tracked']

        b_acc = ckp['best_acc']

        # Print model infos
        self.logger.info(
                    f'Loaded model checkpoint: {model_checkpoint}'
                    f'\n\t\t\tAccuracy: {b_acc}'
                    f'\n\t\t\tNumber of model parameters: {self.__get_total_number_of_training_parameters()}'
                )
            
    def freeze_params(self) -> None:
        """Freeze all layers of the model except the final classifier.

        """
        # Set to False the requires_grad flag for all layers
        for n, param in self.model.named_parameters():
            if 'classifier_1' not in n:
                param.requires_grad = False
            if 'BatchNorm2d' in n:
                m.momentum = 0.0

    def get_model(self) -> nn.Module:
        """Return the model
        
        Returns
        -------
        model : nn.Module
            The model initialized so far

        """
        return self.model
