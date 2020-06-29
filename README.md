# Meta-Learning One-Class Classification with DeepSets: Application in the Milky Way

This repository is the official implementation of Meta-Learning One-Class Classification with DeepSets: Application in the Milky Way

The repository is organized as follow: the Code folder contains the necessary code to reproduce baseline results with Random Forest (folder Random Forest) and the code for generating the data, training and evaluating the Meta-DeepSet model, in folder Deep Sets. The Data folder the data for the Training, Validation and Test streams. Each sub-folder has 2 folders : Stream and 'Background' (i.e. foreground, non-stream stars), which contains the respective stars in separate file for each 'stream problem', with matching ids. 

## Requirements

To install requirements, navigate to the appropriate folder and then run:

```setup
pip install -r requirements.txt
```
## Generate Datasets for Deep Sets

The training and evaluation of the Meta-Deep Sets is done on a dataset format as follow: each element of the dataset is a triplet of (x (the example to classify), y(its labels), s (a support set of positive examples taken as input of the Deep Sets as well) ). To generate these datasets, navigate to "Code/Deep Sets/" and run (for train, validation, or test) the script 'generate_datasets_deepsets.py". 

Use 'mode' argument to generate train, valid, or test dataset. 
Use 'save_as' to indicate where to save the generated pickle files.
Other arguments / parameters used to generate the training set (only):  'foreground_sample_ratio' indicate to which ratio sample the foreground when generating the  dataset, 'factor_positiv_ratio' is used in train to duplicate positive examples, 'max_support_size' and 'min_support_size' are used to cap the support set size, 'max_ratio_support' is an additional cap expressed as a percentage of stream stars to use to sample the support set.
Not using these parameters will generate a similar training set used in the paper:

```generate datasets
python3 generate_datasets_deepsets.py --mode=train --save_as="../../Data/Deep Sets Files/training_test.pkl"
```

## Training

To train the Deep Sets model, navigate to "Code/Deep Sets/" and run 'train_deepset.py'. 
Arguments are: 'num_epochs' (number of iteration to train for --no early stopping--, each iteration covers 'max_count' examples) default 50, 'max_count' (number of examples seen per epoch, default 300000), lr (learning rate, default 1e-3), l1 (l1-regularization, default 1e-6), weight_imbalance (weight to balance the loss, default 100), net_dim (hidden size of the deepset network, default 100), 'train_dataset' and 'eval_dataset' name files for the pickles files, 'log_dir' directory for the logs.
Defaults for max_count, epochs, net_dim are the hyper-parameters used in the paper. 

```train
python3 train_deepset.py --weight_imbalance=1 --train_dataset="../../Data/Deep Sets Files/training_test.pkl" --eval_dataset="../../Data/Deep Sets Files/eval_set.pkl"
```

To train the Recursive Random Forest model, navigate to "Code/Random Forests/" and run:

For the simulated Streams:          Recursive_RF_Simulated_Streams.ipynb

For GD1:                            Recursive_RF_GD1.ipynb

For Pal5 (exploratory):             Recursive_RF_Pal5.ipynb

## Evaluation

To evaluate a trained Deep Sets model on the set of testing streams, navigate to "Code/Deep Sets/" and run 'eval_model.py'. 
Arguments: 'model_file' to indicate the model to load, 'net_dim' the hidden dimension of the Deep Sets to load (default 100), 'eval_dataset' to indicate which file to evaluate on.

We provide the pre-trained models corresponding to the results shown on the Synthetic streams in the paper in the Logs folder.

```eval
python3 eval_model.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params
--log_file='../../Logs/eval_test_final.txt' --eval_dataset=../../Data/Deep Sets Files/test_final_test.pkl
```

Note that evaluation on the testing streams for the Random Forest also takes place within the same Jupyter notebooks.

## Fine-Tuning on GD1

To fine-tune a pretrained Deep Sets model on GD1, navigate to "Code/Deep Sets/" and run 'fine_tune_gd1.py'. The code generates a dataset of similar format. 
'model_file' and 'net_dim' indicate which model to use and hidden dim of the model. 'factor_trainset' indicates the number of negative examples to train on relative to number of positive examples (known) each epoch (resampled from the full negative training set). 'lr' change the learning rate (default is 0.0005)

```fine-tune
python3 fine_tune_gd1.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params
```

## Results

Our models achieve the following performance on :

### GD1

| Model name               |  Precision  |  Recall  |  Balanced Acc  |   MCC   |
| ------------------------ |------------ | -------- | -------------- | ------- |
| DS FT (best train Rec)   |    0.341    |   0.981  |     0.984      |  0.574  |
| Random Forest (Self-Lab) |    0.402    |   0.869  |     0.930      |  0.587  |


