# Meta-Learning One-Class Classification with DeepSets: Application in the Milky Way

This repository is the official implementation of Meta-Learning One-Class Classification with DeepSets: Application in the Milky Way

## Requirements

To install requirements, navigate to the appropriate folder and then run:

```setup
pip install -r requirements.txt
```

## Training

To train the Deep Sets model, navigate to "Code/Deep Sets/" and run (with any additional arguments):

```train
python3 train_deepset.py --max_count=10000 --num_epochs=50
```

To train the Recursive Random Forest model, navigate to "Code/Random Forests/" and run:
For the simulated Streams:          Recursive_RF_Simulated_Streams.ipynb
For GD1:                            Recursive_RF_GD1.ipynb
For Pal5 (exploratory):             Recursive_RF_Pal5.ipynb

Note that evaluation on the testing streams also takes place within the same Jupyter notebooks.

## Evaluation

To evaluate the Deep Sets model on the set of testing streams, navigate to "Code/Deep Sets/" and run:

```eval
python3 eval_model.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params
--log_file='../../Logs/eval_test_final.txt' --eval_dataset=../../Data/Deep Sets Files/test_final_test.pkl
```

## Fine-Tuning on GD1

To fine-tune a pretrained Deep Sets model on GD1, navigate to "Code/Deep Sets/" and run:

```fine-tune
python3 fine_tune_gd1.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params
```

## Results

Our models achieve the following performance on :

### GD1

| Model name               |  Precision  |  Recall  |  Balanced Acc  |   MCC   |
| ------------------------ |------------ | -------- | -------------- | ------- |
| DS FT (best train Rec)   |    0.402    |   0.869  |     0.984      |  0.574  |
| Random Forest (Self-Lab) |    0.341    |   0.981  |     0.930      |  0.587  |


