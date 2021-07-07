# MAFER: a Multi-resolution Approach to Facial Expression Recognition

This repository contains the code relative to the paper "[MAFER: a Multi-resolution Approach to FacialExpression Recognition](...)" by Fabio Valerio Massoli (ISTI - CNR), Donato Cafarelli (ISTI - CNR), Claudio Gennaro (ISTI - CNR), Giuseppe Amato (ISTI - CNR), and Fabrizio Falchi (ISTI - CNR).

We propose a multi-resolution two-step training procedure for Deep Learning models tasked with the Facial Expression Recognition (FER) objective. We prove the benefits of such an approach by extensively test our models on several publicly available datasets.  

**Please note:** 
We are researchers, not a software company, and have no personnel devoted to documenting and maintaing this research code. Therefore this code is offered "AS IS". Exact reproduction of the numbers in the paper depends on exact reproduction of many factors, including the version of all software dependencies and the choice of underlying hardware (GPU model, etc). Therefore you should expect to need to re-tune your hyperparameters slightly for your new setup.


## Proposed Approach

The image below shows the multi-resolution training phase that represents the first step of our learning procedure for FER.

<p align="center">
<img src="https://github.com/fvmassoli/mafer-multires-facial-expression-recognition/blob/main/images/model_simplified.png"  alt="model simplified" width="1000" height="160">
</p>

The confusion matrices below report the performance of our models on the Oulu-CASIA dataset. The quoted numbers are the accuracies, 10-fold averaged, for each expression class.

<p align="center">
<img src="https://github.com/fvmassoli/mafer-multires-facial-expression-recognition/blob/main/images/confusion_matrices.png"  alt="confusion matrices" width="1000" height="220">
</p>

Below we report the t-SNE embedding of deep representations produced by the base model (leftmost) and by the models trained with a multi-resolution training. 

<p align="center">
<img src="https://github.com/fvmassoli/mafer-multires-facial-expression-recognition/blob/main/images/tsne_all.png"  alt="tsne" width="1000" height="220">
</p>

For more details loot at our paper: "[MAFER: a Multi-resolution Approach to FacialExpression Recognition](...)"

## How to run the code

Before to run the code, make sure that your system has the proper packages installed. You can have a look at the [requirements.txt](https://github.com/fvmassoli/mafer-multires-facial-expression-recognition/blob/main/requirements.txt) file.


### Model Train

```
python main_fer_rafdb.py -tr -bp <path_to_base_model> -dn <dataset_name> -df <path_to_dataset>
```

The base_model in this case refers to the original Se-ResNet-50 (model [here](https://cnrsc-my.sharepoint.com/personal/fabrizio_falchi_cnr_it/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9jbnJzYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9mYWJyaXppb19mYWxjaGlfY25yX2l0L0V0bTRiRzFPTjJ0SHF0b202NjZtbWlNQmdPeU9fRnNEd1hJWmgySk9TRlhab3c%5FcnRpbWU9WVhWcUVPc0EyVWc&id=%2Fpersonal%2Ffabrizio%5Ffalchi%5Fcnr%5Fit%2FDocuments%2FSharedByLilnk%2Fpaper%5Fcheckpoints%2Fsenet50%5Fft%5Fpytorch%2Ept&parent=%2Fpersonal%2Ffabrizio%5Ffalchi%5Fcnr%5Fit%2FDocuments%2FSharedByLilnk%2Fpaper%5Fcheckpoints)) from Cao et al. ([paper](https://arxiv.org/abs/1710.08092), [github repo](https://github.com/ox-vgg/vgg_face2)).   

### Model Test
To test the models we use two different scripts. A former one, to test models on a single test set (fer2013 and rafdb datasets), and a latter one, to test models over k-folds (oulucasia dataset).

To test models on the fer2013 or the rafdb datasets:
```
python test_model_fer_raf.py -bp <base_model_ckp> -ck <model_ckp> -dn <dataset_name> -df <dataset_folder> -bs <batch_size>
```

To test models on the oulucasia dataset:
```
python test_model_oulu.py -bp <base_model_ckp> -ck <models_main_folder> -dn <dataset_name> -df <dataset_folder> -bs <batch_size>
```


### Models Checkpoints

All models' checkpoints are available [here](https://github.com/fvmassoli/mafer-multires-facial-expression-recognition/releases/tag/v1.0-fer2013-rafdb).

**FER2013** 
Model | Accuracy (%)
--- | ---
base           | 60.82 
CR             | 73.06
CR-Simiplified | 72.33
CR+AffWild2    | 73.45


**RAF-DB**
Model | Overall Acc. (%) | Average Acc. (%)
--- | --- | ---
base           | 77.09 | 65.39 ± .10
CR             | 88.43 | 81.90 ± .04
CR-Simplified  | 88.14 | 83.16 ± .03
CR+AffWild2    | 88.07 | 82.40 ± .04


**OULU-Casia**

The linked folders cotain all the 10(-fold) models' checkpoints.

Model | Acc. 10-fold avg.
--- | --- 
base          | 59.84 ± 1.29
CR            | 98.40 ± .11
CR-Simplified | 96.72 ± .24
CR+AffWild2   | 98.95 ± .15 


### Dataset mean

The table below reports the arrays for the mean values that we use in our study to center the input data.

Dataset | Mean
--- | --- 
FER-2013 | [133.05986, 133.05986, 133.05986]
RAF-DB | [102.15835, 114.51117, 146.58075]
OULU-Casia | [131.07828, 131.07828, 131.07828]


## Reference
For all the details about the training procedure and the experimental results, please have a look at the [paper](https://arxiv.org/abs/2105.02481).

To cite our work, please use the following form

```
@article{massoli2021mafer,
  title={MAFER: a Multi-resolution Approach to Facial Expression Recognition},
  author={Massoli, Fabio Valerio and Cafarelli, Donato and Gennaro, Claudio and Amato, Giuseppe and Falchi, Fabrizio},
  journal={arXiv preprint arXiv:2105.02481},
  year={2021}
}
```

## Contacts
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

Have fun! :-D
