import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import glob
from astropy.table import Table
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
import math

import argparse

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
# action

parser.add_argument('--factor_trainset', type=int, default=30, help='factor_trainset')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

parser.add_argument('--weight_imbalance', type=float, default=100, help='Imbalance weight to use')
parser.add_argument('--net_dim', type=int, default=100, help="size of hidden layer in deepset")

parser.add_argument('--model_file', type=str, help="Model parameters to load")

parser.add_argument('--log_dir', type=str, default='../../Logs/', help="Log directory")


def eval_loss(model, train_dataset, criterion):
    
    counts = 0
    sum_acc = 0 
    losses = 0

    truth = []
    preds = []

    model.eval()
    with torch.no_grad():
        for i, (x, y, ref) in enumerate(np.random.permutation(train_dataset)):
            #counts += len(y)
            if i==0:
                print(x)
            counts = counts + 1 
            reference = torch.cuda.FloatTensor(ref)
            X = torch.cat((reference, torch.cuda.FloatTensor(x).unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
            X = X.reshape((1, X.size()[0], X.size()[1]))
            #print(X.size())
            X = Variable(X)
            #print(y)
            Y = Variable(torch.cuda.LongTensor([y]))
            
            f_X = model(X)

            preds.append(f_X.max(dim=1)[1].detach().cpu())

            loss = criterion(f_X, Y)
            loss_val = loss.data.cpu().numpy()#[0]
            losses = losses + loss_val
        #     print(" y {}, pred {}".format(y,f_X.detach().cpu().numpy()))
        #     print(loss_val)
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()

            

            y_cut = 1 if y > 0.5 else 0
            truth.append(y_cut)

            del X,Y,f_X,loss
    #         if counts == max_count:
    #             break

        prec, rec, f1, _ = metrics.precision_recall_fscore_support(truth, preds, average="binary")

    #print("Epoch {} : Loss {}, Acc {}, Prec {}, Rec {}".format(e, losses/counts, sum_acc/counts, prec, rec))
    return losses, sum_acc, prec, rec



def train(model, train_dataset, criterion, optimizer):
    
    counts = 0
    sum_acc = 0 
    losses = 0

    truth = []
    preds = []

    model.train()
    for i, (x, y, ref) in enumerate(np.random.permutation(train_dataset)):
        #counts += len(y)
        if i==0:
            print(x)
        counts = counts + 1 
        reference = torch.cuda.FloatTensor(ref)
        X = torch.cat((reference, torch.cuda.FloatTensor(x).unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
        X = X.reshape((1, X.size()[0], X.size()[1]))
        #print(X.size())
        X = Variable(X)
        #print(y)
        Y = Variable(torch.cuda.LongTensor([y]))
        optimizer.zero_grad()
        f_X = model(X)
        
        preds.append(f_X.max(dim=1)[1].detach().cpu())
        
        loss = criterion(f_X, Y)
        loss_val = loss.data.cpu().numpy()#[0]
        losses = losses + loss_val
    #     print(" y {}, pred {}".format(y,f_X.detach().cpu().numpy()))
    #     print(loss_val)
        sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()

        loss.backward()
        deepsets_zaheer.clip_grad(model, 5)
        optimizer.step()
        
        y_cut = 1 if y > 0.5 else 0
        truth.append(y_cut)
        
        del X,Y,f_X,loss
#         if counts == max_count:
#             break
        
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(truth, preds, average="binary")
 
    #print("Epoch {} : Loss {}, Acc {}, Prec {}, Rec {}".format(e, losses/counts, sum_acc/counts, prec, rec))
    return losses, sum_acc, prec, rec, preds, truth

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
            
            softmaxfs = F.softmax(f_X, dim=1)
            preds_probval.append(softmaxfs[0][1].cpu().numpy())
            
            loss = criterion(f_X, Y)
            loss_val = loss.data.cpu().numpy()#[0]
            losses = losses + loss_val
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()

            del X,Y,loss
            if counts % 10000==0:
                print("Count {}".format(counts))
    
    return preds, truth, preds_probval

def eval_criterions(truthes, preds, names, verbose=False):
    sum_prec_normnoise = 0
    sum_rec_normnoise = 0
    sum_f1_normnoise = 0
    sum_mcc_normnoise = 0
    sum_f2_normnoise = 0
    sum_f05_normnoise = 0
    sum_ba_normnoise = 0

    for i in range(len(truthes)):
        ## COmpute prec, recall, f1 for each
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(truthes[i], 
                                                                   preds[i], average="binary")

        sum_prec_normnoise += prec
        sum_rec_normnoise += rec
        sum_f1_normnoise += f1

        ## f-beta?
        _,_,f2, _ = metrics.precision_recall_fscore_support(truthes[i],
                                                            preds[i], average="binary", beta=2)
        _,_,f05, _ = metrics.precision_recall_fscore_support(truthes[i], 
                                                             preds[i], average="binary", beta=0.5)

        sum_f2_normnoise += f2
        sum_f05_normnoise += f05

        ## Compute MCC
        mcc = metrics.matthews_corrcoef(truthes[i], preds[i])
        sum_mcc_normnoise += mcc

        ## Compute Balanced Accuracy
        ba = metrics.balanced_accuracy_score(truthes[i], preds[i])
        sum_ba_normnoise += ba

    div = len(truthes)
    
    if verbose:
        print("Averaged Prec {}, Rec {}, F1 {}".format(sum_prec_normnoise / div, sum_rec_normnoise /div, 
                                                       sum_f1_normnoise/div))
        print("Averaged F2 {} F0.5 {}".format(sum_f2_normnoise/div, sum_f05_normnoise/div))
        print("Averaged MCC {}".format(sum_mcc_normnoise / div))
        print("Averaged BA {}".format(sum_ba_normnoise/div))
    return sum_prec_normnoise / div,  sum_rec_normnoise /div, sum_f1_normnoise/div, \
             sum_f2_normnoise/div, sum_f05_normnoise/div, sum_mcc_normnoise / div, sum_ba_normnoise/div


def normalize(dataframe_tot, labels_not_normalized):
    # (x - u) / s
    features_to_normalize = list( set(dataframe_tot.columns) - set(labels_not_normalized))
    dataframe = dataframe_tot[features_to_normalize]
    
    if np.any(np.isin(dataframe.std(),0)):
        if dataframe.columns[np.isin(dataframe.std(),0)] != 'is_known':
            print("std is 0, which is weird ?")
            print(dataframe.std())
            print(dataframe)
        std = dataframe.std()
        std[np.isin(std,0)]=1.
        dataNorm=(dataframe-dataframe.mean())/std
    else:
        dataNorm=(dataframe-dataframe.mean())/dataframe.std()
        
    dataNorm[labels_not_normalized]=dataframe_tot[labels_not_normalized]
    return dataNorm


if __name__ == '__main__':


    args = parser.parse_args()


    ## Load model 


    # df_results = pd.read_csv('../../Logs_to_evaluate/df_run_0.csv')
    #f = df_results.iloc[df_results[args.criterion_val].idxmax()]['file']
    ## 
    file_param = args.model_file
    f = file_param

    network_dim = args.net_dim # int(f.split("_")[10])
    is_D5 = True #(f.split("_")[5] == "D5") # D / D5
    

    x_dim = 10 * 2 
    device = 'cuda'

    if is_D5:
        model = deepsets_zaheer.D5(network_dim, x_dim=x_dim, pool='mean').cuda()
    else: 
        model =  deepsets_zaheer.D(network_dim, x_dim=x_dim, pool='mean').cuda()

    model.load_state_dict(torch.load(file_param))
    model.eval()


    ## Load data 
    ## Some of this should be moved somewhere else to just load....
    gd1_pd = pd.read_csv('../../Data/Deep Sets Files/master_gd1_rev2.csv')

    features_wspec = ["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g", 'species', 'in_gd1']
    gd1_reduced_spec = gd1_pd[features_wspec]
    gd1_reduced_spec = gd1_reduced_spec.drop_duplicates(["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g"])
    
    gd1_reduced_spec = gd1_reduced_spec.reset_index(drop=True)


    features = ["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g"]
    gd1_reduced = gd1_reduced_spec[features]
    in_gd1 = gd1_reduced_spec['in_gd1']
    species = gd1_reduced_spec['species']

    gd1_selflab = pd.read_csv('../../Data/Test Set/GD1/testing_gd1.csv')

    species_to_gd1 = np.zeros(len(species))
    species_to_gd1[species=='stream_train'] = 1
    species_to_gd1[species=='stream_test'] = 1


    
    features = ["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g"]
    gd1_streamtrain = gd1_reduced[species=='stream_train'][features]
    gd1_streamtrain['label'] = 1
    gd1_streamtrain['is_ref'] = 1
    gd1_streamtrain['self_lab'] = 0

    features = ["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g"]
    gd1_noisetrain = gd1_reduced[species=='noise_train'][features]
    gd1_noisetrain['label'] = 0
    gd1_noisetrain['is_ref'] = 2
    gd1_noisetrain['self_lab'] = 0

    features = ["ra", "dec", "pmra_x", "pmdec_x", "g_bp", "g_rp", "g"] #'ra', 'dec', 'pmra', 'pmdec', 'g', 'g_bp', 'g_rp', 'ang0', 'ang1', 'ang2'
    gd1_only_test = gd1_selflab[gd1_selflab['self_lab']==False][features]
    gd1_only_test['is_ref'] = 0
    gd1_only_test['label'] = 0
    gd1_only_test.loc[gd1_selflab['stream_mask']==True, 'label'] =1
    gd1_only_test['self_lab'] = 2 
    #gd1_only_test['self_lab'][gd1_selflab['self_lab']==False] = 2



    gd1_streamtrain= gd1_streamtrain.rename(columns={'pmra_x':'pmra', 'pmdec_x':'pmdec'})
    gd1_noisetrain= gd1_noisetrain.rename(columns={'pmra_x':'pmra', 'pmdec_x':'pmdec'})
    gd1_only_test= gd1_only_test.rename(columns={'pmra_x':'pmra', 'pmdec_x':'pmdec'})


    
    all_data = pd.concat([gd1_streamtrain, gd1_noisetrain, gd1_only_test], ignore_index=True, sort=False)#, gd1_noisetrain))

    ## Add angular coord: radian = degree*(pi/180)
    dec_rad = all_data['dec'].values * (math.pi / 180.)
    ra_rad = all_data['ra'].values * (math.pi / 180.)
    pmdec = all_data['pmdec'].values
    pmra = all_data['pmra'].values

    all_data['ang0'] = - pmdec * np.cos(ra_rad) - pmra * np.cos(dec_rad) * np.sin(dec_rad) * np.sin(ra_rad)
    all_data['ang1'] = - pmra * np.cos(dec_rad) * np.cos(ra_rad) * np.sin(dec_rad) + pmdec * np.sin(ra_rad)
    all_data['ang2'] = pmra * np.cos(dec_rad) * np.cos(dec_rad)

    all_data_normlz = normalize(all_data, ['label', 'is_ref', 'self_lab'])

    ## Generate training set positive only from gd-1

    features =  ['ra', 'dec', 'pmra', 'pmdec', 'g_bp', 'g_rp', 'g', 'ang0', 'ang1', 'ang2']

    all_data_posittrain = all_data_normlz[all_data_normlz['is_ref']==1]
    all_data_posittrain = all_data_posittrain.reset_index(drop=True)
    posittrain_np  = all_data_posittrain[features].to_numpy()
    posittrain_label = np.ones(posittrain_np.shape[0])


    test_set = all_data_normlz[all_data_normlz['is_ref']==0]
    test_set = test_set.reset_index(drop=True)
    test_set_numpy = test_set[features].to_numpy()

    
    data_to_eval = []

    ys_true_final = []
    for i in range(len(test_set)):
        if  (test_set['self_lab'][i] == 2):
            if (test_set['label'][i] == 1 ):
                data_to_eval.append((test_set_numpy[i,:], 1., posittrain_np))
                ys_true_final.append(1.)
            else: 
                data_to_eval.append((test_set_numpy[i,:], 0., posittrain_np))
                ys_true_final.append(0)
        else:
            print("PROBLEM IN DATA ???")

    
    label_modifier = 0
    fake_noiselabel = np.zeros(len(gd1_noisetrain)) + label_modifier #+ 0.1
    

    all_data_noisetrain = all_data_normlz[all_data_normlz['is_ref']==2][features].to_numpy()
    

    args = parser.parse_args()

    factor_trainset = args.factor_trainset

    idx_small_neg = random.sample(range(all_data_noisetrain.shape[0]), posittrain_np.shape[0] * factor_trainset)

    merged_posselflab = np.append(posittrain_np, all_data_noisetrain[idx_small_neg], axis=0)
    print(merged_posselflab.shape)
    merged_labels = np.append(posittrain_label, fake_noiselabel[idx_small_neg], axis=0)
    print(merged_labels.shape)
    pool_supportset = posittrain_np #np.append(posittrain_np, all_data_selflab[np.where(np.array(preds_valselflab) > 0.5 )[0],:],
                                #axis=0)
    print(pool_supportset.shape)



    tot_support = pool_supportset.shape[0]
    number_ex = merged_posselflab.shape[0] #tot_support * 10
    min_sup = 5
    max_sup = 40

    training_set_gd1 = []


    for j in range(number_ex):
        ## Sample a support set betwen min_sup and max_sup
        idx_support = random.sample(range(tot_support), random.randint(min_sup, max_sup))
        support_set = pool_supportset[idx_support,:]
        training_set_gd1.append( (merged_posselflab[j], merged_labels[j], support_set) )

    
    only_noise_gd1 = []
    for j in range(all_data_noisetrain.shape[0]):
        idx_support = random.sample(range(tot_support), random.randint(min_sup, max_sup))
        support_set = pool_supportset[idx_support,:]
        only_noise_gd1.append((all_data_noisetrain[j], 0., support_set))

    all_gd1 = only_noise_gd1
    

    ## Build a separate only_true_support dataset to evaluate only on this

    only_true_support = []
    for j in range(posittrain_np.shape[0]):
        only_true_support.append( (posittrain_np[j], 1., np.append(posittrain_np[:j], posittrain_np[j+1:], axis=0)))
        all_gd1.append( (posittrain_np[j], 1., np.append(posittrain_np[:j], posittrain_np[j+1:], axis=0)))
    ## Build only noisetrain dataset

    new_lr = args.lr
    optimizer = optim.Adam([{'params':model.parameters()}], lr=new_lr, weight_decay=1e-4)

    # filename = './logs_to_evaluate/onlysup2_saveparam_noselflabel_{}_l10.4_{}_modiflab_{}.txt'.format(factor_trainset,new_lr, 
    #                 label_modifier)

    writer = SummaryWriter('{}/finetuneGD1_{}_{}_{}_{}_{}'.format( args.log_dir, new_lr,
                     args.net_dim, factor_trainset, label_modifier, 1e-4))

    modelname =  './{}/params_D5finetuneGD1_{}_{}_{}{}_{}'.format( args.log_dir, new_lr,
                     args.net_dim, factor_trainset, label_modifier, 1e-4)

    

    weight_imbalance = 1
    weight_class = torch.FloatTensor([1, weight_imbalance]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_class).cuda()


    losses, sum_acc, prec, rec = eval_loss(model,training_set_gd1, criterion)

    writer.add_scalar('training loss', losses, -1)
    writer.add_scalar('training prec', prec, -1)
    writer.add_scalar('training rec', rec, -1)
    
    print("Loss Init {}, Prec init {}, Rec init {}".format(losses, prec, rec))
    
     

    max_epoch_finetune = int(len(gd1_noisetrain) / ( len(gd1_streamtrain) * factor_trainset)) * 3
    print("Max epoch : {}".format(max_epoch_finetune))


    best_train_f1 = 0.0
    best_train_f2  = 0.0
    best_train_f05 = 0.0
    best_train_recall = 0.0

    for i in range( max_epoch_finetune):

        print("Iter {}".format(i))
        model.train()
        losses, sum_acc, prec, rec, preds_train, truth_train = train(model,training_set_gd1, criterion, optimizer) #

        print("Iter {}, on 'training': loss {}, prec {}, rec {}".format(i, losses, prec, rec))

        writer.add_scalar('training loss', losses, i)
        writer.add_scalar('training prec', prec, i)
        writer.add_scalar('training rec', rec, i)
        
        ## Compute criterions on entire training_set with noisy labels 0
        
        
        ## Compute prec, rec etc on only_true_support
        preds, truth, preds_val = get_preds_truths(model, only_true_support, criterion)

        print(len(preds))
        print(len(truth))
        val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba = eval_criterions([truth],
                                                                                  [preds],
                                                                                  ["gd1 original true suport set"])

        print("Iter {}: true support set only : prec {:.3f} & rec {:.3f} & f1 {:.3f} & f2 {:.3f} & f05 {:.3f}  &  mcc {:.3f} & ba {:.3f}\n".format(
             i, val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba))
        
        writer.add_scalar('true support set prec', val_prec, i)
        writer.add_scalar('true support set rec', val_prec, i)
        writer.add_scalar('true support set f1', val_f1, i)
        writer.add_scalar('true support set f2', val_f2, i)
        writer.add_scalar('true support set f05', val_f05, i)
        writer.add_scalar('true support set mcc', val_mcc, i)
        writer.add_scalar('true support set ba', val_ba, i)

        # posnoise_preds = preds_noisetrain + preds
        # posnoise_truths = truth_noisertain + truth
        preds_noisetrain, truth_noisertain, preds_noisertainval = get_preds_truths(model, all_gd1, criterion)

        val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba = eval_criterions([truth_noisertain],
                                                                                  [preds_noisetrain],
                                                                                  ["gd1 full training set"])

        print("Iter {}: training set : prec {:.3f} & rec {:.3f} & f1 {:.3f} & f2 {:.3f} & f05 {:.3f}  &  mcc {:.3f} & ba {:.3f}".format(
             i, val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba))

        writer.add_scalar('full training set prec', val_prec, i)
        writer.add_scalar('full training set rec', val_prec, i)
        writer.add_scalar('full training set f1', val_f1, i)
        writer.add_scalar('full training set f2', val_f2, i)
        writer.add_scalar('full training set f05', val_f05, i)
        writer.add_scalar('full training set mcc', val_mcc, i)
        writer.add_scalar('full training set ba', val_ba, i)

        ## If best in train for diff criterion, save model
        if val_f1 > best_train_f1:
            torch.save(model.state_dict(), "{}_best_f1.params".format(modelname))
            best_train_f1 = val_f1
        if val_f2 > best_train_f2:
            torch.save(model.state_dict(), "{}_best_f2.params".format(modelname))
            best_train_f2 = val_f2
        if val_f05 > best_train_f05:
            torch.save(model.state_dict(), "{}_best_f05.params".format(modelname))
            best_train_f05 = val_f05
        if val_rec > best_train_recall:
            torch.save(model.state_dict(), "{}_best_rec.params".format(modelname))
            best_train_recall = val_rec


        ## Resample dataset with new negative examples        
        idx_small_neg = random.sample(range(all_data_noisetrain.shape[0]), posittrain_np.shape[0] * factor_trainset)

        merged_posselflab = np.append(posittrain_np, all_data_noisetrain[idx_small_neg], axis=0)
        merged_labels = np.append(posittrain_label, fake_noiselabel[idx_small_neg], axis=0)
        
        pool_supportset = posittrain_np 
                                       

        tot_support = pool_supportset.shape[0]
        number_ex = merged_posselflab.shape[0] #tot_support * 10
        min_sup = 5
        max_sup = 40

        training_set_gd1 = []
        for j in range(number_ex):
            ## Sample a support set betwen min_sup and max_sup
            idx_support = random.sample(range(tot_support), random.randint(min_sup, max_sup))
            support_set = pool_supportset[idx_support,:]
            training_set_gd1.append( (merged_posselflab[j], merged_labels[j], support_set) )

        

        ## Compute all losses on test_final
        preds, truth, preds_val = get_preds_truths(model,data_to_eval, criterion)       
        val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba = eval_criterions([truth],
                                                                                  [preds],
                                                                                  ["gd1 test-final"])

        print("Iter {}: final test : prec {:.3f} & rec {:.3f} & f1 {:.3f} & f2 {:.3f} & f05 {:.3f}  &  mcc {:.3f} & ba {:.3f}".format(
             i, val_prec, val_rec, val_f1, val_f2, val_f05, val_mcc, val_ba))

        
        writer.add_scalar('final test set prec', val_prec, i)
        writer.add_scalar('final test set rec', val_prec, i)
        writer.add_scalar('final test set f1', val_f1, i)
        writer.add_scalar('final test set f2', val_f2, i)
        writer.add_scalar('final test set f05', val_f05, i)
        writer.add_scalar('final test set mcc', val_mcc, i)
        writer.add_scalar('final test set ba', val_ba, i)

# python3 fine_tune_gd1.py --model_file=../../Logs/best_models/deepset_D5_1_0.001_100_50_300000_1e-06_bestF1.params