import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import glob
import pickle as pkl
import os
import matplotlib.pyplot as plt
import random
import math

def normalize(dataframe, labels_not_normalized):
    # (x - u) / s
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
    dataNorm[labels_not_normalized]=dataframe[labels_not_normalized]
    return dataNorm

## Generate a training dataset
## Returns a list of triplets [instance, label, support set]
## Noise sample ratio : how many negative (foreground) stars are sampled as a ratio to stream's stars
## Factor positive ratio  : factor to duplicate the positive examples
## Max support size : maximum size of the support set sampled 
## Max ratio support : maximum size of the support set expressed as ratio regarding size of the stream
## Min support size : minimum size of the support set sampled
## final support set is sampled within stream's stars and its size is sampled between 
## (min_support_size , min(max_ratio_support * size_stream, max_support_size)
## 
## Not activated : Known_ratio : inject false negative from the stream. if 1 = no false negative, uses all streams'stars for positive

def setup_train_dataset(noise_sample_ratio = 150,                   
                      verbose = False, factor_positive_ratio = 2,
                      max_support_size = 150, min_support_size = 5,
                     max_ratio_support = 0.5, folder_stream ='../../Data/Training Set/Training Streams/Stream Stars',
                     folder_foreground='../../Data/Training Set/Training Streams/Background Stars'):

    train = [] 
    group_a_names = ['stream-1324','stream-1424','stream-1864', 'stream-1954','stream-2471','stream-2985',
                     'stream-3132','stream-4106','stream-4175','stream-4394','stream-4468','stream-4677',
                     'stream-4990','stream-5711','stream-575','stream-5793','stream-5797','stream-5830',
                     'stream-5896','stream-5921','stream-5982','stream-6446','stream-6839','stream-7700',
                     'stream-847','stream-8482','stream-8735','stream-9164','stream-9420','stream-1664',
                     'stream-411','stream-5911','stream-2022','stream-8947','stream-1827','stream-6049', 
                     'stream-4325','stream-1591','stream-5344','stream-52','stream-7595','stream-1655',
                     'stream-9971','stream-9466','stream-8874','stream-9390']
    
    stream_known_cnt = 0
    support_set_sizeslist = {}
    
    working_stream = 0 
    total_pos_examples = 0
    total_neg_examples = 0
    total_stream_stars = 0
            
    for i in range(len(group_a_names)):
        stream_name = group_a_names[i]

        # Reading in streams and filtering on b
        
        stream = pd.read_csv("{}/{}_intelligent_cut.csv".format(folder_stream, stream_name), index_col=0)
        
        print("{} number of stars after cut: {}".format(stream_name, stream.values.shape[0]))
        
        if(stream.values.shape[0] < 10):
            print("{} ignored because not enough stars (less than 10) after cut: {} ".format(stream_name, 
                                                                                stream.values.shape[0]))
        
        else:
            working_stream = working_stream + 1
            # Calculating color variables for each stream
            stream["g"] = stream["phot_g_mean_mag"]
            stream["g_bp"] = stream["phot_g_mean_mag"] - stream["phot_bp_mean_mag"]
            stream["g_rp"] = stream["phot_g_mean_mag"] - stream["phot_rp_mean_mag"]

            # Reading in respective noise points for each stream
            noise = pd.read_csv('{}/{}_mul_150_total_noise.csv'.format(folder_foreground, stream_name), index_col=None)
            
            noise["g"] = noise["phot_g_mean_mag"]
            noise["g_bp"] = noise["phot_g_mean_mag"] - noise["phot_bp_mean_mag"]
            noise["g_rp"] = noise["phot_g_mean_mag"] - noise["phot_rp_mean_mag"]

            features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g"]
            noise = noise[features]
            stream = stream[features]

            # Labelling each set as stream or noise
            stream["is_stream_actual"] = 1
            noise["is_stream_actual"] = 0

            # Setting up all known points
            
            
            ## For a given stream : 
            ##      Generate M (M = number of streams stars * noise_sample_ratio) 
            ##             negative samples with shuffled support set
            ##      Generate N (number of stars in the streams) * factor_positive positive samples w/ 
            ##         shuffled support set
            
            nb_stream_stars = len(stream)
            number_positive_ex = nb_stream_stars * factor_positive_ratio
            number_negative_ex = nb_stream_stars * noise_sample_ratio
            
            if number_negative_ex > len(noise):
                print("Weird thing where number of noise points is not 1:150 ?")
                number_negative_ex = len(noise)
                print("Number of noise examples : {}, number of stream stars post cut: {}".format(
                len(noise), len(stream)))
            
            ## Add angular coord and normalize first, then loop and generate


            ## concat everything to compute ang coord and normalize
            all_data = pd.concat((stream, noise), sort=False, ignore_index=True)

            ## Add angular coord: radian = degree*(pi/180)
            dec_rad = all_data['dec'].values * (math.pi / 180.)
            ra_rad = all_data['ra'].values * (math.pi / 180.)
            pmdec = all_data['pmdec'].values
            pmra = all_data['pmra'].values

            all_data['ang0'] = - pmdec * np.cos(ra_rad) - pmra * np.cos(dec_rad) * np.sin(dec_rad) * np.sin(ra_rad)
            all_data['ang1'] = - pmra * np.cos(dec_rad) * np.cos(ra_rad) * np.sin(dec_rad) + pmdec * np.sin(ra_rad)
            all_data['ang2'] = pmra * np.cos(dec_rad) * np.cos(dec_rad)

            all_data_normlz = normalize(all_data, ["is_stream_actual"])

            features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g", "ang0", "ang1", "ang2"]
            
            stream_normalized = all_data_normlz[all_data_normlz["is_stream_actual"]==1][features].to_numpy()
            noise_normalized = all_data_normlz[all_data_normlz["is_stream_actual"]==0][features].to_numpy()
            
            
            smallest_ref_size = 100000000
            largest_ref_size = 0
            
            support_sizes = []
            ## Loop to generate the examples:
            
            
            for neg_ex in range(number_negative_ex):
                ## Sample a support set of a random size withing a range : 

                size_support_set = random.randint(  min_support_size, 
                                                   min( int(max_ratio_support*nb_stream_stars), max_support_size))
                
                index_support = random.sample(list(np.arange(nb_stream_stars)), size_support_set)
                reference_points = stream_normalized[index_support,:].copy()
                train.append((noise_normalized[neg_ex], 0, reference_points.copy()))

                smallest_ref_size = min(smallest_ref_size, reference_points.shape[0])
                largest_ref_size = max(largest_ref_size, reference_points.shape[0])   
                support_sizes.append(reference_points.shape[0])
            
            for pos_ex in range(number_positive_ex):
                ## if factor_positive_ratio > 1 we need to sample:
                idx_pos = random.randint(0, nb_stream_stars-1)
  

                ## Sample a support set of a random size withing a range : here I choose a min support size 3?
                size_support_set = random.randint(  min_support_size, 
                                                   min( int(max_ratio_support*nb_stream_stars), max_support_size))
                
                ## remove star observed from possible support set
                list_of_indexes = list(np.arange(nb_stream_stars))
                list_of_indexes.remove(idx_pos)
                
                index_support = random.sample(list_of_indexes, size_support_set)
                reference_points = stream_normalized[index_support,:].copy()
                train.append((stream_normalized[idx_pos], 1, reference_points.copy()))
                
                smallest_ref_size = min(smallest_ref_size, reference_points.shape[0])
                largest_ref_size = max(largest_ref_size, reference_points.shape[0])   
                support_sizes.append(reference_points.shape[0])
                
            support_set_sizeslist[stream_name] = support_sizes
            if verbose == True:
                    print(stream_name)
                    print('Reference point min size: {}'.format(smallest_ref_size))
                    print('Reference point max size: {}'.format(largest_ref_size))
                    print('Supervision point count: {}'.format(number_positive_ex))
                    print('Noise point count: {}'.format(number_negative_ex))
                    
            total_pos_examples = total_pos_examples + number_positive_ex
            total_neg_examples = total_neg_examples + number_negative_ex
            total_stream_stars = total_stream_stars + nb_stream_stars
            print("--"*10)
            

    print('Number of train streams: {}'.format(working_stream))
#     print('Smallest reference set: '+str(smallest_ref_size))
#     print('Largest reference set: '+str(largest_ref_size))    
    print('Number of streams stars : {}, number of pos examples : {}'.format(total_stream_stars,
                                                                             total_pos_examples))
    print("Number of negative examples total : {}".format(total_neg_examples))
    print('Number of train points: '+ str(len(train)))
    
    return train, support_set_sizeslist

# The following function sets up the classification and reference sets for our Test streams
# Return 2 lists of N lists (N = number of streams),
# each of the N lists is composed of triplets [instance, label, support set]
# First list is the pool of self-labeling examples, 2d list is the 'final' tes set 

## TEST :
## group_c_names = ['stream-5402','stream-1101','stream-1519','stream-247', 'stream-2805','stream-4717',
##                     'stream-5713','stream-9528']
## stream_cut_folder = 'stream_stars_split/group_c_intelligent/'
## self_lab_split_file = ./half_test/testing_{}.csv


## VALID :
## group_c_names = group_b_names = ['stream-1012','stream-1667','stream-1698', 'stream-178','stream-3775','stream-5489',
##                    'stream-8137']
## stream_cut_folder = 'stream_stars_split/group_b_intelligent/'
## self_lab_split_file = ./half_test/validation_{}.csv


def setup_test_dataset_half_groupstreams(group_c_names, stream_cut_folder, self_lab_split_file):

    test_selflab_complete = []
    test_final_complete = []
    
    
    # Missing noise for 'stream-4807'
    stream_known_cnt = 0
    smallest_ref_size = 100000000
    largest_ref_size = 0    
    
    for stream_name in group_c_names:
        #print(stream_name)
        test_final = []
        test_selflab = []
        # Read in known and unknown stream points from group_c folder
        stream_known = pd.read_csv("{}/{}_known_to_model.csv".format(stream_cut_folder,stream_name),index_col=0)
        
#         stream_known = stream[stream['self_lab'] == False].copy()
    
        test_data = pd.read_csv('{}_{}.csv'.format(self_lab_split_file, stream_name))
        
        # Calculating color variables for each known stream point
        stream_known["g"] = stream_known["phot_g_mean_mag"]
        stream_known["g_bp"] = stream_known["phot_g_mean_mag"] - stream_known["phot_bp_mean_mag"]
        stream_known["g_rp"] = stream_known["phot_g_mean_mag"] - stream_known["phot_rp_mean_mag"]

        # Reading in respective noise points for each stream
        #noise = pd.read_csv('stream_stars_split/simulated_noise_points/ratios_intelligent/'+stream_name+'_mul_150_total_noise.csv', index_col=None)
        test_data["g"] = test_data["phot_g_mean_mag"]
        test_data["g_bp"] = test_data["phot_g_mean_mag"] - test_data["phot_bp_mean_mag"]
        test_data["g_rp"] = test_data["phot_g_mean_mag"] - test_data["phot_rp_mean_mag"]
        
        print(stream_name)
        
        # Labelling each set as stream or noise
        # and if half-test or not....

        # Setting up reference set of stream points       
        stream_known["is_reference"] = 1
        test_data["is_reference"] = 0
        
        # Labelling points and creating distinct reference set    
        test_data["label"] = 0
        #test_data.loc[:, ('label', test_data['stream_mask']==True)] = 1
        test_data["label"][test_data['stream_mask']==True]=1
        
        test_data["half_test"] = 0
        #test_data.loc[:, ('half_test', test_data['self_lab']==False)] = 1
        test_data["half_test"][test_data['self_lab']==False] = 1
        
        #stream_unknown["label"] = 1
        stream_known["half_test"] = 2
        stream_known["label"] = 1
        
        features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g", "is_reference", "label", "half_test"]
        test_data = test_data[features]
        stream_known = stream_known[features]
        
        nb_noise_total = np.sum(test_data['label']==0)
        nb_stream_total = np.sum(test_data['label']==1) + len(stream_known)
        nb_stream_unk = np.sum(test_data['label']==1)
        print('noise count: '+str(nb_noise_total))
        print("stream count in test total {}".format(nb_stream_unk))
        
        print('stream-to-noise ratio: '+str(nb_noise_total/(nb_stream_total)))
        
        ## Final_noise = background, label=0, is_ref = 0
        ## Stream_known = stream used as support set , label = 1, is_ref = 1
        ## stream_unknown = set used to evaluate output, "is_reference"=0, label = 1

        ## concat everything to normalize
        all_data = pd.concat((stream_known,test_data), sort=False, ignore_index=True)
        ## Add angular coord: radian = degree*(pi/180)
        dec_rad = all_data['dec'].values * (math.pi / 180.)
        ra_rad = all_data['ra'].values * (math.pi / 180.)
        pmdec = all_data['pmdec'].values
        pmra = all_data['pmra'].values
        
        all_data['ang0'] = - pmdec * np.cos(ra_rad) - pmra * np.cos(dec_rad) * np.sin(dec_rad) * np.sin(ra_rad)
        all_data['ang1'] = - pmra * np.cos(dec_rad) * np.cos(ra_rad) * np.sin(dec_rad) + pmdec * np.sin(ra_rad)
        all_data['ang2'] = pmra * np.cos(dec_rad) * np.cos(dec_rad)
        
        all_data_normlz = normalize(all_data, ["label", "is_reference", "half_test"])
        
        features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g", "ang0", "ang1", "ang2"]
        
        
        
        #features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g"]
        reference_points = all_data_normlz[all_data_normlz.is_reference==1][features].to_numpy()
        
        #final_noise = noise.sample(len(stream_known[stream_known.is_reference == 1])*noise_sample_ratio)
        #final_noise = noise_with_stream_noise.sample(int(len(noise_with_stream_noise)*noise_sample_ratio))
        
        # reference_points = stream_known.copy()
        smallest_ref_size = min(smallest_ref_size,len(reference_points))
        largest_ref_size = max(largest_ref_size,len(reference_points))              
        stream_known_cnt += nb_stream_unk
        
#         Creating classification dataset with final noise (unknown stream points and actual noise) and known stream points
#         classification_dataset = pd.concat((noise.copy(), stream_unknown.copy()), sort=False, ignore_index = True)    
#         features = ["ra", "dec", "pmra", "pmdec", "g_bp", "g_rp", "g"]
#         reference_points = reference_points[features].to_numpy()
#         X = classification_dataset[features].to_numpy()
#         y = classification_dataset["label"].to_numpy()

        classification_dataset = all_data_normlz[all_data_normlz.is_reference==0].copy()
        X = classification_dataset[features].to_numpy()
        y = classification_dataset["label"].to_numpy()
        half_test_label = classification_dataset["half_test"].to_numpy()

        # Appending the reference set to each classification point
        count_selflab = 0
        count_final = 0
        nb_str_selflab = 0
        nb_str_final = 0
        nb_noise_selflab = 0
        nb_noise_final = 0
        for j in range(X.shape[0]):
            if half_test_label[j] == 0:
                test_selflab.append((X[j].copy(), y[j].copy(), reference_points.copy()))
                count_selflab = count_selflab + 1
                if y[j] == 1:
                    nb_str_selflab += 1
                else:
                    nb_noise_selflab += 1
                
            elif  half_test_label[j] == 1:
                test_final.append((X[j].copy(), y[j].copy(), reference_points.copy()))
                count_final = count_final + 1
                if y[j] == 1 :
                    nb_str_final += 1
                else:
                    nb_noise_final += 1
            else:
                print("This shouldn't happen ?")

        print("{}, support set {}, self-lab size total {}, final size total {}".format(stream_name, 
                                                len(reference_points), count_selflab, count_final))
        print(" {} streams, {} noise, in self-lab ".format(nb_str_selflab, nb_noise_selflab))
        print(" {} streams, {} noise, in final ".format(nb_str_final, nb_noise_final))
        print("---"*5)
        
        test_selflab_complete.append(test_selflab)
        test_final_complete.append(test_final)
        
    print('Number of test streams: '+str(len(group_c_names)))
    print('Smallest reference set: '+str(smallest_ref_size))
    print('Largest reference set: '+str(largest_ref_size))    
    print('Number of known stream points (non-ref): '+str(stream_known_cnt))
    print('Number of test points in self-lab: '+ str(len(test_selflab)))      
    print('Number of test points in final lab: '+ str(len(test_final)))      
    
    return test_selflab_complete, test_final_complete

import argparse

parser = argparse.ArgumentParser()
# action

parser.add_argument('--mode', type=str, help="Which dataset to generate: [train, valid, test]")
# parser.add_argument('--known_ratio', type=float, default=1.0, 
#     help='Known ratio parameter used to generate training set. Default is 1 --no false negative--')
parser.add_argument('--foreground_sample_ratio', type=int, default=150, 
    help = 'Ratio to sample foreground in training. Default --and max -- is 150. Use data mul400 for higher ratio.')
parser.add_argument('--factor_positive_ratio', type=int, default=2, 
    help = 'Factor to duplicate positive (stream) examples. Default is 2.')
parser.add_argument('--max_support_size', type=int, default = 150,
    help = 'Maximum size of support set when generating training examples. Default is 150.')
parser.add_argument('--min_support_size', type=int, default = 7,
    help = 'Minimum size of support set when generating training examples. Default is 7')
parser.add_argument('--max_ratio_support', type=float, default = 0.5,
    help = 'Maximum percentage of stream stars used to sample support set for train set. Default is 0.5')

parser.add_argument('--save_as', type=str, help="Where to save dataset")


if __name__ == '__main__':

    args = parser.parse_args()

    if args.mode == 'train':
        print("Generating training dataset")
        folder_stream = '../../Data/Training Set/Training Streams/Stream Stars'
        folder_foreground = '../../Data/Training Set/Training Streams/Background Stars'
        train_set, _ = setup_train_dataset( #known_ratio = args.known_ratio, 
                    noise_sample_ratio = args.foreground_sample_ratio, 
                      verbose = True, factor_positive_ratio = args.factor_positive_ratio,
                      max_support_size = args.max_support_size, min_support_size = args.min_support_size,
                     max_ratio_support = args.max_ratio_support, folder_stream = folder_stream, 
                     folder_foreground = folder_foreground)

        pkl.dump(train_set, open(args.save_as, 'wb'))


    elif args.mode == 'valid':

        print("Generating Meta Validation datasets : self-lab set and finale-test set")
        group_c_names =  ['stream-1012','stream-1667','stream-1698', 'stream-178','stream-3775',
        'stream-5489',  'stream-8137']
        stream_cut_folder = '../../Data/Training Set/Validation Streams/Stream Stars'
        self_lab_split_file = '../../Data/Test Set/Validation Streams/validation'

        self_lab, test = setup_test_dataset_half_groupstreams(group_c_names, stream_cut_folder, self_lab_split_file)
        pkl.dump(self_lab, open(args.save_as + 'self_lab.pkl', 'wb'))
        pkl.dump(test, open(args.save_as + '_final_test.pkl', 'wb'))


    elif args.mode == 'test':

        print("Generating Meta Test datasets : self-lab set and finale-test set")
        group_c_names = ['stream-5402','stream-1101','stream-1519','stream-247', 'stream-2805','stream-4717',
                         'stream-5713','stream-9528']
        stream_cut_folder = '../../Data/Training Set/Evaluation Streams/Stream Stars'
        self_lab_split_file = '../../Data/Test Set/Evaluation Streams/testing'
        
        self_lab, test = setup_test_dataset_half_groupstreams(group_c_names, stream_cut_folder, self_lab_split_file)
        pkl.dump(self_lab, open(args.save_as + 'self_lab.pkl', 'wb'))
        pkl.dump(test, open(args.save_as + '_final_test.pkl', 'wb'))

    else:
        print("invalid argument for mode")


## python3 generate_datasets_deepsets.py --mode=train --save_as="../../Data/Deep Sets Files/training_test.pkl"
## python3 generate_datasets_deepsets.py --mode=valid --save_as="../../Data/DeepSets Files/valid"
## python3 generate_datasets_deepsets.py --mode=test --save_as="../../Data/DeepSets Files/test"
