import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import os
import matplotlib.pyplot as plt
import random

import sklearn.metrics as metrics

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import copy

import deepsets_zaheer

import argparse
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
# action

parser.add_argument('--weight_imbalance', type=float, default=100, help='Imbalance weight to use')
parser.add_argument('--net_dim', type=int, default=100, help="size of hidden layer in deepset")

parser.add_argument('--model_file', type=str, help="Model parameters to load")
parser.add_argument('--eval_dataset', type=str, default='../../Data/Deep Sets Files/valid_final_test.pkl', help="Validation dataset")

parser.add_argument('--log_file', type=str, default='../../Logs/eval.txt', help="Log directory")

    

def get_preds_truths(model,val_dataset, criterion):
    counts = 0
    sum_acc = 0
    losses = 0
    model.eval()
    truth = []
    preds = []
    preds_probval = []
    with torch.no_grad():
        for i, (x, y, ref) in enumerate(val_dataset):
            counts = counts + 1 
            reference = torch.cuda.FloatTensor(ref)
            X = torch.cat((reference, torch.cuda.FloatTensor(x).unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
            X = X.reshape((1, X.size()[0], X.size()[1]))
            X = Variable(X)
            Y = Variable(torch.cuda.LongTensor([y]))

            f_X = model(X)
            truth.append(y)
            preds.append(f_X.max(dim=1)[1].detach().cpu().numpy())
            
            softmaxfs = F.softmax(f_X)
            preds_probval.append(softmaxfs[0][1].cpu().numpy())
            
            loss = criterion(f_X, Y)
            loss_val = loss.data.cpu().numpy()#[0]
            losses = losses + loss_val
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()

            del X,Y,loss
            
    
    return preds, truth, preds_probval, losses

def eval_criterions(truths, preds, names, verbose=False):
    sum_prec_normnoise = 0
    sum_rec_normnoise = 0
    sum_f1_normnoise = 0
    sum_mcc_normnoise = 0
    sum_f2_normnoise = 0
    sum_f05_normnoise = 0
    sum_ba_normnoise = 0
    for i in range(len(truths)):
        ## Compute prec, recall, f1 for each
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(truths[i], 
                                                                   preds[i], average="binary")
        
        sum_prec_normnoise += prec
        sum_rec_normnoise += rec
        sum_f1_normnoise += f1

        _,_,f2, _ = metrics.precision_recall_fscore_support(truths[i],
                                                            preds[i], average="binary", beta=2)
        _,_,f05, _ = metrics.precision_recall_fscore_support(truths[i], 
                                                             preds[i], average="binary", beta=0.5)

        sum_f2_normnoise += f2
        sum_f05_normnoise += f05


        ## Compute MCC
        mcc = metrics.matthews_corrcoef(truths[i], preds[i])
        sum_mcc_normnoise += mcc

        ## Compute Balanced Accuracy
        ba = metrics.balanced_accuracy_score(truths[i], preds[i])
        sum_ba_normnoise += ba
        
    div = len(truths)
    
    if verbose:
        print("Averaged Prec {}, Rec {}, F1 {}".format(sum_prec_normnoise / div, sum_rec_normnoise /div, 
                                                       sum_f1_normnoise/div))
        print("Averaged F2 {} F0.5 {}".format(sum_f2_normnoise/div, sum_f05_normnoise/div))
        print("Averaged MCC {}".format(sum_mcc_normnoise / div))
        print("Averaged BA {}".format(sum_ba_normnoise/div))


    return sum_prec_normnoise / div,  sum_rec_normnoise /div, sum_f1_normnoise/div, \
             sum_f2_normnoise/div, sum_f05_normnoise/div, sum_mcc_normnoise / div, sum_ba_normnoise/div


if __name__ == '__main__':

    args = parser.parse_args()

    device = 'cuda'

    val_dataset = pickle.load(open(args.eval_dataset,"rb"))
    val_names = np.arange(len(val_dataset))

    weight_im = args.weight_imbalance
    current_netdim = args.net_dim
    
    if weight_im < 1:
        weight_class = torch.FloatTensor([1./weight_im, 1]).to(device)
    else:
        weight_class = torch.FloatTensor([1, weight_im]).to(device)



    criterion = torch.nn.CrossEntropyLoss(weight=weight_class,reduction='sum').cuda()  



    network_dim = current_netdim
    x_dim = 10 * 2 
    model = deepsets_zaheer.D5(network_dim, x_dim=x_dim, pool='mean').cuda()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()


    all_truths_normnoise = []
    all_preds_normnoise = []
    all_preds_value_normnoise = []
    val_loss = 0.
    count = 0.
    for i in range(len(val_dataset)):
        preds, truth, preds_val, loss_val = get_preds_truths(model,val_dataset[i], criterion)
        all_truths_normnoise.append(truth)
        all_preds_normnoise.append(preds)
        all_preds_value_normnoise.append(preds_val)
        val_loss += loss_val
        count += len(val_dataset[i])
    val_loss = val_loss / count
    
    val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba = eval_criterions(all_truths_normnoise,
                                                                              all_preds_normnoise,
                                                                              val_names)
    
    print("Eval on {} : prec {:.3f} & rec {:.3f} & f1 {:.3f} & f2 {:.3f} & f05 {:.3f}  &  mcc {:.3f} & ba {:.3f}".format(
        args.eval_dataset, val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba))

    f = open(args.log_file, 'w')
    f.write("Eval on {} : prec {:.3f} & rec {:.3f} & f1 {:.3f} & f2 {:.3f} & f05 {:.3f}  &  mcc {:.3f} & ba {:.3f}".format(
        args.eval_dataset, val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba))
    f.close()

# python3 eval_model.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params
# --log_file='../../Logs/eval_test_final.txt' --eval_dataset=../../Data/Deep Sets Files/test_final_test.pkl 
