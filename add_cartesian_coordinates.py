import filter_isochrone as fi
import numpy as np
from astropy.table import Table
import glob

stream_data_base = 'gaia_mock_streams/'
stream_files = glob.glob(stream_data_base + 'stream*.fits.gz')

#!mkdir "gaia_mock_streams_cartesian"

for index, stream_file in enumerate(stream_files):
    print("Stream file ", str(index), " ", stream_file)
    
    table = Table.read(stream_file, format='fits')
    stream = table.to_pandas()
    print("Original length of stream: ", len(stream))
    if len(stream) == 0:
        continue
    #create features 'g_bp' and 'g_rp' that are g - bp and g - rp, respectively
    stream['g_bp'] = stream['phot_g_mean_mag'] - stream['phot_bp_mean_mag']
    stream['g_rp'] = stream['phot_g_mean_mag'] - stream['phot_rp_mean_mag']
    #create feature 'g' based on 'phot_g_mean_mag'
    stream['g'] = stream['phot_g_mean_mag']

    log_age, distances_std_dev, distances = fi.optimize_age_metallicity(stream)

    #returns distance modulus
    distance_moduli = fi.isochrone_filter_distances(stream, age = log_age[0], feh = log_age[1], only_ms_rgb = True)
    print("length of distance moduli: ", len(distance_moduli))
    #distance moduli = 5 * log (r/10) --> 10*e^(distance_moduli / 5)
    r = 10 * np.exp(distance_moduli / 5)

    #now we have distance from earth, so we can
    #convert to cartesian coordinates
    deg2rad = lambda deg: deg/180.*np.pi

    theta = deg2rad(90 - stream['dec'])
    phi = deg2rad(stream['ra'])
    ra = deg2rad(stream['ra'])
    dec = deg2rad(stream['dec'])

    stream['x'] = r*np.sin(theta)*np.cos(phi)
    stream['y'] = r*np.sin(theta)*np.sin(phi)
    stream['z'] = r*np.cos(theta)
    # All of these are in kiloparsecs
    cos = np.cos
    sin = np.sin
    pmra = stream['pmra']
    pmdec = stream['pmdec']

    stream['vx'] = r * (cos(ra)*cos(dec)*pmdec - sin(ra)*sin(dec)*pmra)
    stream['vy'] = r * (cos(ra)*sin(dec)*pmra  + cos(dec)*sin(ra)*pmdec)
    stream['vz'] = r * (sin(dec)*pmdec)
    
    index_slash = stream_file.find('/')
    index_dot = stream_file.find('.')
    new_filename = stream_file[:index_slash] + "_cartesian" + stream_file[index_slash:index_dot] + ".csv"
    print("length of new stream: ", stream)
    print("new file name: ", new_filename)
    print()
    stream.to_csv(new_filename)
