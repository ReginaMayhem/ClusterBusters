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
import copy

import torch.optim as optim
from torch.autograd import Variable

import deepsets_zaheer

import argparse
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, default=50, help='num epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--l1', type=float, default=1e-6, help='L1 regul.')
parser.add_argument('--weight_imbalance', type=float, default=100, help='Imbalance weight to use')
parser.add_argument('--net_dim', type=int, default=100, help="size of hidden layer in deepset")
parser.add_argument('--max_count', type=int, default=300000, help="Number of examples seen in one epoch, eval after max_count")

parser.add_argument('--train_dataset', type=str, default='../../Data/Deep Sets Files/training_test.pkl', help="Training dataset")
parser.add_argument('--eval_dataset', type=str, default='../../Data/Deep Sets Files/valid_final_test.pkl', help="Validation dataset")

parser.add_argument('--log_dir', type=str, default='../../Logs/', help="Log directory")

def train(model, train_dataset, criterion, optimizer, max_count=1000):
    
    counts = 0
    sum_acc = 0 
    losses = 0

    truth = []
    preds = []

    model.train()
    for i, (x, y, ref) in enumerate(np.random.permutation(train_dataset)):
        
        counts = counts + 1 
        reference = torch.cuda.FloatTensor(ref)
        X = torch.cat((reference, torch.cuda.FloatTensor(x).unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
        X = X.reshape((1, X.size()[0], X.size()[1]))
        X = Variable(X)
        Y = Variable(torch.cuda.LongTensor([y]))
        optimizer.zero_grad()
        f_X = model(X)
        
        truth.append(y)
        preds.append(f_X.max(dim=1)[1].detach().cpu())
        
        loss = criterion(f_X, Y)
        loss_val = loss.data.cpu().numpy()#[0]
        losses = losses + loss_val

        sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()

        loss.backward()
        deepsets_zaheer.clip_grad(model, 5)
        optimizer.step()
        del X,Y,f_X,loss
        if counts == max_count:
            break
        
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(truth, preds, average="binary")
 
    
    return losses, sum_acc, prec, rec
    

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

    train_dataset = pickle.load(open(args.train_dataset, "rb"))
    val_dataset = pickle.load(open(args.eval_dataset,"rb"))
    val_names = np.arange(len(val_dataset))

    print(len(train_dataset))

    current_l1 = args.l1 
    weight_im = args.weight_imbalance
    current_lr = args.lr 
    current_netdim = args.net_dim
    
    if weight_im < 1:
        weight_class = torch.FloatTensor([1./weight_im, 1]).to(device)
    else:
        weight_class = torch.FloatTensor([1, weight_im]).to(device)



    criterion = torch.nn.CrossEntropyLoss(weight=weight_class,reduction='sum').cuda()  



    network_dim = current_netdim
    x_dim = 10 * 2 
    model = deepsets_zaheer.D5(network_dim, x_dim=x_dim, pool='mean').cuda()


    num_epochs = args.num_epochs
    
    optimizer = optim.Adam([{'params':model.parameters()}], lr=current_lr, weight_decay=current_l1) #, eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, min_lr=1e-7)
                


    max_count = args.max_count


    writer = SummaryWriter('{}/D5_{}_{}_{}_{}_{}_{}'.format( args.log_dir, weight_im, current_lr,
                     current_netdim, args.num_epochs, args.max_count, current_l1))


    model_name = "./{}/params_D5_{}_{}_{}_{}_{}_{}".format( args.log_dir, weight_im, current_lr,
                 current_netdim, args.num_epochs, args.max_count, current_l1)
    


    max_val_loss = float('inf')
    crits = ['loss','prec','rec', 'f1', 'f2']#,'accu']
    valid_losses = {}
    valid_losses['loss'] = float('inf')
    valid_losses['prec'] = float('inf') * -1
    valid_losses['rec'] = float('inf') * -1
    valid_losses['f1'] = float('inf') * -1
    valid_losses['f2'] = float('inf') * -1


    for e in range(num_epochs):
        

        losses, sum_acc, prec, rec = train(model, train_dataset, criterion, optimizer, max_count)

        #scheduler.step()
        print("Epoch {} : Loss {}, Prec {}, Rec {}".format(e, losses/max_count,  prec, rec))
        
        writer.add_scalar('training loss', losses/max_count, e)
        writer.add_scalar('training prec', prec, e)
        writer.add_scalar('training rec', rec, e)

        
        scheduler.step(losses)


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
        
        print("E{} Valid : Loss {}, Prec {}, Rec {}, F1 {}".format(e, val_loss, val_prec, val_rec, val_f1))
        
        writer.add_scalar('valid loss', val_loss, e)
        writer.add_scalar('valid prec', val_prec, e)
        writer.add_scalar('valid rec', val_rec, e)
        writer.add_scalar('valid f1', val_f1, e)
        writer.add_scalar('valid f2', val_f2, e)
        writer.add_scalar('valid f05', val_f05, e)
        writer.add_scalar('valid mcc', val_mcc, e)
        writer.add_scalar('valid ba', val_ba, e)

        tmp_val_loss ={}
        tmp_val_loss['loss'] = val_loss
        tmp_val_loss['prec'] = val_prec
        tmp_val_loss['rec'] = val_rec
        tmp_val_loss['f1'] = val_f1
        tmp_val_loss['f2'] = val_f2

        for c in crits:
            if c == 'loss':
                if tmp_val_loss[c] < valid_losses[c]:
                    torch.save(model.state_dict(), "{}_{}.params".format(model_name,c))
                    valid_losses[c] = tmp_val_loss[c]
            else:
                if tmp_val_loss[c] > valid_losses[c]:
                    torch.save(model.state_dict(), "{}_{}.params".format(model_name,c))
                    valid_losses[c] = tmp_val_loss[c]
       



        
        
        
## python3 train_deepset.py --max_count=10000  + all other arguments             
