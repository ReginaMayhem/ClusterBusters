from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy.table import Table

#let us write a function to extract the Gaia noise points based on some inputs
def obtain_noise(min_ra, max_ra, min_dec, max_dec, max_rel_err, n_points):
    
    qry = f" \n\
    select top {n_points} source_id, \n\
    dr2.ra, \n\
    dr2.dec, \n\
    parallax, \n\
    parallax_error, \n\
    pmra, \n\
    pmdec, \n\
    phot_g_mean_mag,\n\
    phot_bp_mean_mag, \n\
    phot_rp_mean_mag, \n\
    bp_rp, \n\
    bp_g, \n\
    g_rp\n\
    from gaiadr2.gaia_source as dr2 \n\
    where dr2.ra > {min_ra} and dr2.ra < {max_ra} and dr2.dec > {min_dec} and dr2.dec < {max_dec} \n\
    and parallax is not null \n\
    and parallax_error is not null \n\
    and abs(dr2.parallax/dr2.parallax_error) < {max_rel_err} \n\
    and pmra is not null \n\
    and pmdec is not null \n\
    and phot_g_mean_mag is not null \n\
    and phot_bp_mean_mag is not null \n\
    and phot_rp_mean_mag is not null \n\
    and bp_rp is not null \n\
    and bp_g is not null \n\
    and g_rp is not null \n\
    order by random_index"

    data_noise = Gaia.launch_job_async(qry).get_results().to_pandas()
    
    return data_noise

def create_stream_dataset(stream_file, min_ra, max_ra, min_dec, max_dec, noise_multiplier, known_stream_stars_ratio, train_noise_ratio, save_path):
    
    table = Table.read(stream_file, format='fits')
    
    stream = table.to_pandas()
    
    stream = stream.query(f'ra > {min_ra} & ra < {max_ra} & dec < {max_dec} & dec > {min_dec}')
    
    arr = np.zeros(len(stream))
    arr[np.random.choice(np.arange(len(stream)), replace=False, size = int(known_stream_stars_ratio*len(stream)))] = 1
    stream = stream.assign(is_train = arr, is_stream = 1)

    
    
    noise = obtain_noise(min_ra, max_ra, min_dec, max_dec, 0.5, noise_multiplier*len(stream))
    
    arr = np.zeros(len(noise))
    arr[np.random.choice(np.arange(len(noise)), replace=False, size = int(np.sum(stream["is_train"])*train_noise_ratio))] = 1
    
    noise = noise.assign(is_train = arr, is_stream = 0)
    df = pd.concat((noise, stream), axis=0, ignore_index=True, sort=True)
    print(len(stream))
    print(np.sum(stream["is_train"]))
    print(len(noise))
    print(np.sum(noise["is_train"]))
    df.to_csv(save_path, index=False)
    
    
    

    
    