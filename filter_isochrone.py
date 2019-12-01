import pandas as pd
import numpy as np
import pickle as pkl
from astropy import units as u
import astropy.coordinates as coord
from sklearn.neighbors import NearestNeighbors
import itertools
from tqdm import tqdm
feh_possible = [-4., -3.5, -3., -2.5, -2., -1.75, -1.5, -1.25, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5]
log_age_possible = [5., 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6., 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7., 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8., 8.05, 8.1, 8.15, 8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95, 9., 9.05, 9.1, 9.15, 9.2, 9.25, 9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75, 9.8, 9.85, 9.9, 9.95, 10., 10.05, 10.1, 10.15, 10.2, 10.25, 10.3]

def load_raw_isochrones():
    """Return isochrones as pandas object"""
    with open('/scratch/kae358/Capstone/isochrone_filter_new/all_gaia_isochrones.pkl', 'rb') as f:
        data = pkl.load(f)
    return data

def filter_isochrones(isochrones, age=None, feh=None, only_ms_rgb=None):
    
    if only_ms_rgb:
        EEP = isochrones['EEP'].copy()
        eep_cut = (EEP > 202) & (EEP < 454)
        #eep_cut = (EEP > 202) & (EEP < 605)
        isochrones = isochrones[eep_cut].copy()

    new_isochrones = isochrones[[col for col in 'G BP RP log_age feh_init'.split(' ')]].copy()
    new_isochrones.rename(columns={"G": "g", "BP": "bp", "RP": "rp"}, inplace=True)

    if age is not None: 
        log_age = np.log10(age)
        best_log_age = find_nearest_1d(log_age_possible, log_age)
        new_isochrones = new_isochrones[np.abs(new_isochrones['log_age'] - best_log_age) < 0.025]

    if feh is not None: 
        best_feh = find_nearest_1d(feh_possible, feh)
        new_isochrones = new_isochrones[np.abs(new_isochrones['feh_init'] - best_feh) < 0.25/2.]

    for color in ['bp', 'rp']:
        new_isochrones['g_' + color] = new_isochrones['g'] - new_isochrones[color]

    return new_isochrones
def load_isochrones(age=None, feh=None, only_ms_rgb=False):
    isochrones = load_raw_isochrones()

    new_isochrones = filter_isochrones(isochrones, age, feh, only_ms_rgb)
    
    return new_isochrones


def find_nearest_1d(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def isochrone_filter_distances(points, age=None, feh=None, only_ms_rgb=False, isochrone=None):
    if isochrone is None:
        isochrone = load_isochrones(age=age, feh=feh, only_ms_rgb=only_ms_rgb)
    else:
        isochrone = filter_isochrones(isochrone, age=age, feh=feh, only_ms_rgb=only_ms_rgb)

    g_points = np.array(isochrone['g'])
    g_bp_points = isochrone['g_bp']
    g_rp_points = isochrone['g_rp']
    point_set = np.array([g_bp_points, g_rp_points]).T
    isochrone_nn_tree = NearestNeighbors(n_neighbors=1).fit(point_set)

    magnitude_color = np.array(points[['g_bp', 'g_rp']].copy())

    _, nearest_neighbor_gbp_grp = isochrone_nn_tree.kneighbors(magnitude_color)

    how_far_is_point_from_isochrone = np.array(points['g']) - g_points[nearest_neighbor_gbp_grp[:, 0]]
    return how_far_is_point_from_isochrone


def isochrone_filter(points, distance, allowed_distance, only_ms_rgb=False,
                     age=None, feh=None, isochrones=None):

    how_far_is_point_from_isochrone = isochrone_filter_distances(
            points, age, feh, only_ms_rgb, isochrone=isochrones)
    effective_distance_from_isochrone = 10*u.pc* (10**((how_far_is_point_from_isochrone)/5))

    return np.abs(effective_distance_from_isochrone - distance) < allowed_distance

def split_data_pd(data_boxes, splits):
    #For example, :
    # splits = [(col, 10) for col in [0, 1, 2]]
    # data_splits = split_data([data], splits)

    if len(splits) == 0:
        return data

    each_X_data = []

    split_column = splits[0][0]
    split_size = splits[0][1]

    for data in data_boxes:

        splitting_data = np.array(data[split_column])
        indices = np.argsort(splitting_data)
        cur_splits = np.array_split(indices, split_size)
        for cur_split in cur_splits:
            each_X_data.append(data.iloc[cur_split])


    if len(splits) == 1:
        return each_X_data
    else:
        return split_data_pd(each_X_data, splits[1:])

def optimize_age_metallicity(points):
    
    best = (0,0)
    best_distance_diff = 99999
    isochrones_raw = load_raw_isochrones()
    combs = list(itertools.product(log_age_possible, feh_possible))
    for log_age, feh in tqdm(combs):
        distances_from_isochrones = isochrone_filter_distances(points, 10**log_age, feh, True, isochrones_raw)
        #distance_diff = max(distances_from_isochrones) - min(distances_from_isochrones)
        distance_diff = np.std(distances_from_isochrones)
        if distance_diff < best_distance_diff:
            best_distance_diff = distance_diff
            best = (log_age, feh)
            
    distances_from_isochrones = isochrone_filter_distances(points, 10**best[0], best[1], True, isochrones_raw)
    effective_distances = 10*u.pc* (10**((how_far_is_point_from_isochrone)/5))
    return best, best_distance_diff_log, effective_distances_from_best
        