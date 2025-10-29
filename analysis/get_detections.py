import h5py as h5
import numpy as np
from scipy.integrate import quad

import astropy.units as u
import astropy.constants as c

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from importlib import reload

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import cogsworth
from StroopPop import StroopPop
from gala import dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from legwork import evol, utils, source

import os
import argparse

import time
import sys
# sys.path.append("../src/")
from compas_processing import mask_COSMIC_data, get_COSMIC_vars
from galaxy import simulate_mw, simulate_simple_mw
from variations import variations

np.random.seed()

def save_initC(filename, initC, key="initC", settings_key="initC_settings", force_save_all=False):
    """Save an initC table to an HDF5 file.

    Any column where every binary has the same value (setting) is saved separately with only a single copy
    to save space.

    This will take slightly longer (a few seconds instead of 1 second) to run but will save you around
    a kilobyte per binary, which adds up!

    Parameters
    ----------
    filename : `str`
        Filename/path to the HDF5 file
    initC : `pandas.DataFrame`
        Initial conditions table
    key : `str`, optional
        Dataset key to use for main table, by default "initC"
    settings_key : `str`, optional
        Dataset key to use for settings table, by default "initC_settings"
    force_save_all : `bool`, optional
        If true, force all settings columns to be saved in the main table, by default False
    """

    # for each column, check if all values are the same
    uniques = initC.nunique(axis=0)
    compress_cols = [col for col in initC.columns if uniques[col] == 1]

    if len(compress_cols) == 0 or force_save_all:
        # nothing to compress, just save the whole table
        initC.to_hdf(filename, key=key, mode='a', format='table', append=True)
    else:
        # save the main table without the compressed columns
        initC.drop(columns=compress_cols).to_hdf(filename, key=key, mode='a', format='table', append=True)

        # save the compressed columns separately
        settings_df = pd.DataFrame([{col: initC[col].iloc[0] for col in compress_cols}])
        settings_df.to_hdf(filename, key=settings_key, mode='a', format='table', append=True)

def append_orbits_to_file(filepath, new_orbits):
    # Get total length and offsets of this batch
    new_orbits = [orbit for orbit in new_orbits if len(orbit.pos) > 0]  # filter out empty orbits
    orbit_lengths = [len(orbit.pos) for orbit in new_orbits]
    orbit_lengths_total = sum(orbit_lengths)
    new_offsets = np.insert(np.cumsum(orbit_lengths), 0, 0)

    # Initialize data arrays
    orbits_data = {
        "pos": np.zeros((3, orbit_lengths_total)),
        "vel": np.zeros((3, orbit_lengths_total)),
        "t": np.zeros(orbit_lengths_total),
        "offsets": new_offsets
    }

    for i, orbit in enumerate(new_orbits):
        start = new_offsets[i]
        end = new_offsets[i + 1]
        orbits_data["pos"][:, start:end] = orbit.pos.xyz.to(u.kpc).value
        orbits_data["vel"][:, start:end] = orbit.vel.d_xyz.to(u.km/u.s).value
        orbits_data["t"][start:end] = orbit.t.to(u.Myr).value

    with h5.File(filepath, "a") as f:
        if "orbits" not in f:
            grp = f.create_group("orbits")
            grp.create_dataset("pos", data=orbits_data["pos"], maxshape=(3, None))
            grp.create_dataset("vel", data=orbits_data["vel"], maxshape=(3, None))
            grp.create_dataset("t", data=orbits_data["t"], maxshape=(None,))
            grp.create_dataset("offsets", data=orbits_data["offsets"], maxshape=(None,))
        else:
            grp = f["orbits"]
            old_len = grp["t"].shape[0]
            new_len = old_len + orbit_lengths_total

            # Resize each dataset
            grp["pos"].resize((3, new_len))
            grp["vel"].resize((3, new_len))
            grp["t"].resize((new_len,))
            grp["offsets"].resize((grp["offsets"].shape[0] + len(new_offsets) - 1,))  # skip 1st offset

            # Write appended data
            grp["pos"][:, old_len:new_len] = orbits_data["pos"]
            grp["vel"][:, old_len:new_len] = orbits_data["vel"]
            grp["t"][old_len:new_len] = orbits_data["t"]

            # Update offsets (excluding 0), shifted by old_len
            grp["offsets"][-(len(new_offsets)-1):] = new_offsets[1:] + old_len

parser=argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=2500,
                    help='Number of runs to perform for each binary type')
parser.add_argument('--cores', type=int, default=48,
                    help='Number of cores to use for the runs')
parser.add_argument('--btype', type=str, default='BHBH')
parser.add_argument('--reset', action='store_true', help='Reset the output file before running the script')
parser.add_argument('--model', help = 'Physics variation', default='fiducial')
args = parser.parse_args()

RUNS = args.runs
CORES = args.cores
BTYPE = args.btype
btype = BTYPE


# kickflag = 5 and kickflag = 1
# qcflag = 2 and 5
# ecsn_mlow = 2.6 ecsn = 2.95, and ecsn_mlow = 1.8 ecsn = 2.5
# alpha1 = 1 and 5

# fiducial: kickflag = 5, qcflag = 5, ecsn_mlow = 1.8..., alpha1 = 1

fiducial = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.8, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 5, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 5, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

kick_var = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.8, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 5, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 1, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

qc_var = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.8, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 2, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 5, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

ecsn_var = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.95, 'ecsn_mlow' : 2.6, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 5, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 5, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

alpha_var = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 5.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.8, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 5, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 5, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

if args.model == 'fiducial':
    BSEDict = fiducial
elif args.model == 'kick_var':
    BSEDict = kick_var
elif args.model == 'qc_var':
    BSEDict = qc_var
elif args.model == 'ecsn_var':
    BSEDict = ecsn_var
elif args.model == 'alpha_var':
    BSEDict = alpha_var

old = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 
            'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 
            'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 
            'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
            'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
            'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
            'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.4, 'aic' : 1, 'ussn' : 0, 
            'sigmadiv' :-20.0, 'qcflag' : 2, 'eddlimflag' : 0, 
            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
            'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 
            'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 1, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 
            'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}

# Define the binary types to process
k_select_dict = {'NSNS': ([13],[13]), 'BHNS': ([14], [13]), 'BHBH': ([14],[14]), 'BHWD': ([14],[10,11,12]), 'NSWD': ([13], [10, 11, 12]), 'all': ([10, 11, 12, 13, 14], [10, 11, 12, 13, 14]) }

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set display width to avoid line breaks
pd.set_option('display.max_rows', 500)  # Show more rows if needed

k_select = k_select_dict[btype]
k1_select = k_select[0]
k2_select = k_select[1]

# set filepath
stroopfile = "/hildafs/projects/phy230054p/fep/stroopwafel_dir/stroopwafel/tests/output/{}/{}/{}.h5".format(btype, args.model, btype)
# stroopfile = "/hildafs/projects/phy230054p/fep/stroopwafel_dir/stroopwafel/tests/output/BHBH.h5" # for testing
filepath = "../data/LISA_sources/{}/{}/sample_{}.h5".format(btype, args.model, btype)

# Check if the output directory exists, if not create it
output_dir = os.path.dirname(filepath)
os.makedirs(output_dir, exist_ok=True)

if args.reset:
    with h5.File(filepath, 'w') as f:
        # Create an empty file to reset the output
        pass
    current_run = 0
else:
    # Check current run number
    with h5.File(filepath, 'r') as f:
        current_run = f.attrs.get("run_number", 0)

# read file
initC = pd.read_hdf(stroopfile, key='initC')
weights = pd.read_hdf(stroopfile, key='weights')


# Update the weights in initC
DCO_bin_nums = initC.bin_num.values
DCO_weights = weights.loc[weights.bin_num.isin(DCO_bin_nums)].mixture_weight.values
initC["mixture_weight"] = DCO_weights
weights_sum = initC.mixture_weight.values.sum()


# find galaxy metallicity weights
print("Calculating Milky Way metallicity weights...")
grid = np.linspace(-4, np.log10(0.03), 50)
inner_bins = np.array([grid[i] + (grid[i+1] - grid[i]) / 2 for i in range(len(grid) - 1)])
bins = 10**np.hstack((grid[0], inner_bins, grid[-1]))
bin_centers = np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins) - 1)])
g = cogsworth.sfh.SandersBinney2015(size=10_000_000, potential=gp.MilkyWayPotential2022())
counts, _ = np.histogram(g.Z.value, bins=bins)
MW_weights = counts / np.sum(counts)

# add metallicity weights to initC
initC_bins = np.digitize(initC.metallicity.values, bins) - 1
mapped_weights = np.take(MW_weights, initC_bins)
initC["MW_weights"] = mapped_weights

print("Beginning sample galaxy runs...")
for run in range(current_run + 1, RUNS + current_run + 1):
    time_start = time.time()

    rand_sample = initC.sample(n=200000, replace=True)

    order = np.argsort(rand_sample.metallicity.values)
    rand_sample = rand_sample.iloc[order].reset_index(drop=True)
    weights = rand_sample.mixture_weight.values
    metallicity_weights = rand_sample.MW_weights.values
    weights_sum = weights.sum()
    full_weights_sum = (weights * metallicity_weights).sum()
    print(f"Running {btype} run {run} of {RUNS}")
    # Load the population
    pop = StroopPop(rand_sample, len(rand_sample), processes=CORES, BSE_settings={},
             store_entire_orbits=False, 
                    sfh_model=cogsworth.sfh.SandersBinney2015, 
                    sfh_params={"potential": gp.MilkyWayPotential2022()},
                    galactic_potential=gp.MilkyWayPotential2022())

    # pop = StroopPop(rand_sample, len(rand_sample), processes=CORES, BSE_settings={},
    #          store_entire_orbits=False)
    
    pop.BSE_settings = {}

    print("Sampling initial binaries")
    sample_start = time.time()
    pop.sample_initial_binaries()
    print(f"Time taken to sample binaries: {time.time() - sample_start:.2f} seconds")
    print("Evolving binaries")
    evol_start = time.time()
    pop.perform_stellar_evolution()

    # add weights and old bin_num to initC and final_bpp
    pop.initC['stroop_bin_num'] = rand_sample.bin_num.values
    pop.initC['mixture_weight'] = weights
    pop.initC['MW_weights'] = metallicity_weights
    pop.final_bpp['stroop_bin_num'] = rand_sample.bin_num.values
    pop.final_bpp['mixture_weight'] = weights
    pop.final_bpp['MW_weights'] = metallicity_weights

    sources_mask = pop.final_bpp["sep"] > 0.0
    pairs_mask = ((pop.final_bpp.kstar_1.isin(k1_select)) & (pop.final_bpp.kstar_2.isin(k2_select))) | \
                 ((pop.final_bpp.kstar_1.isin(k2_select)) & (pop.final_bpp.kstar_2.isin(k1_select)))
    full_pairs_mask = ((pop.bpp.kstar_1.isin(k1_select)) & (pop.bpp.kstar_2.isin(k2_select))) | \
                 ((pop.bpp.kstar_1.isin(k2_select)) & (pop.bpp.kstar_2.isin(k1_select)))
    print(len(pairs_mask))
    dco_mask = (pop.bpp["sep"] > 0.0) & full_pairs_mask
    interesting_systems_table = pop.bpp.loc[dco_mask].drop_duplicates(subset = 'bin_num', keep = 'first')
    bin_num_mask = pop.final_bpp.bin_num.isin(interesting_systems_table.bin_num.values)
    dco_weights_sum = np.sum(pop.final_bpp.loc[bin_num_mask].mixture_weight.values)
    full_dco_weights_sum = np.sum(pop.final_bpp.loc[bin_num_mask].mixture_weight.values * pop.final_bpp.loc[bin_num_mask].MW_weights.values)
    print(f"{BTYPE}s formed: ", len(interesting_systems_table))
    print("bpp length: ", len(pop.bpp))
    print("Number of final binaries that don't fit pairs: ", len(pop.final_bpp[~pairs_mask]))

    print(f"Time taken to evolve binaries: {time.time() - evol_start:.2f} seconds")
  
    # throw out binaries that merged or disrupted
    

    pop = pop[sources_mask & pairs_mask]

    GWmask = (2/(pop.final_bpp.porb.values * u.day).to(u.s)) > (1e-6*u.Hz)

    if btype == 'NSWD':
        GWmask = (2/(pop.final_bpp.porb.values * u.day).to(u.s)) > (1e-4*u.Hz)

    pop = pop[GWmask]

    print("Evolving galaxy")

    galaxy_start = time.time()
    pop.perform_galactic_evolution()
    print(f"Time taken to evolve galaxy: {time.time() - galaxy_start:.2f} seconds")


    x, y, z = pop.final_pos.T.value
    pop.final_bpp["x"] = x
    pop.final_bpp["y"] = y
    pop.final_bpp["z"] = z
    pop.final_bpp["run"] = int(run)
    pop.final_bpp["weights_sum"] = weights_sum
    pop.final_bpp["full_weights_sum"] = full_weights_sum # metallicity weights * stroop weights
    pop.final_bpp["dco_weights_sum"] = dco_weights_sum 
    pop.final_bpp["full_dco_weights_sum"] = full_dco_weights_sum 



    sources4 = pop.to_legwork_sources(assume_mw_galactocentric=True)
    sources10 = pop.to_legwork_sources(assume_mw_galactocentric=True)
    sources4.get_snr(t_obs = 4 * u.yr)
    sources10.get_snr(t_obs = 10 * u.yr)

    full_snr4 = sources4.snr
    full_snr10 = sources10.snr


    v_phi = (pop.initial_galaxy.v_T / pop.initial_galaxy.rho)
    v_X = (pop.initial_galaxy.v_R * np.cos(pop.initial_galaxy.phi)
              - pop.initial_galaxy.rho * np.sin(pop.initial_galaxy.phi) * v_phi)
    v_Y = (pop.initial_galaxy.v_R * np.sin(pop.initial_galaxy.phi)
                + pop.initial_galaxy.rho * np.cos(pop.initial_galaxy.phi) * v_phi)
    
    w0s = gd.PhaseSpacePosition(pos=[a.to(u.kpc).value for a in [pop.initial_galaxy.x,
                                                                     pop.initial_galaxy.y,
                                                                     pop.initial_galaxy.z]] * u.kpc,
                                    vel=[a.to(u.km/u.s).value for a in [v_X, v_Y,
                                                                        pop.initial_galaxy.v_z]] * u.km/u.s)
    no_kick_orbits = [pop.galactic_potential.integrate_orbit(w0s[i],
                                                       t1=0 * u.Myr,
                                                       t2=pop.initial_galaxy.tau[i],
                                                       dt=1 * u.Myr,
                                                       Integrator=gi.DOPRI853Integrator) # change to 0.01 Myr
                  for i in range(len(pop))]

    x_no_kick = [no_kick_orbits[i][-1].x.value for i in range(len(no_kick_orbits))]
    y_no_kick = [no_kick_orbits[i][-1].y.value for i in range(len(no_kick_orbits))]
    z_no_kick = [no_kick_orbits[i][-1].z.value for i in range(len(no_kick_orbits))]

    initialx, initialy, initialz = pop.initial_galaxy.x.value, pop.initial_galaxy.y.value, pop.initial_galaxy.z.value
    initialvx, initialvy, initialvz = v_X.value, v_Y.value, pop.initial_galaxy.v_z.value
    finalx, finaly, finalz = pop.final_pos.T.value
    finalvx, finalvy, finalvz = pop.final_vel.T.value

    component = pop.initial_galaxy.which_comp

    # component_map = {
    # 'low_alpha_disc': 0,
    # 'high_alpha_disc': 1,
    # 'bulge': 2}   

    component_map = {
    'thin_disc': 0,
    'thick_disc': 1}

    component = np.vectorize(component_map.get)(component)

    pop.initC["x"] = initialx
    pop.initC["y"] = initialy
    pop.initC["z"] = initialz
    pop.initC["v_x"] = initialvx
    pop.initC["v_y"] = initialvy
    pop.initC["v_z"] = initialvz
    pop.initC["component"] = component
    pop.initC["run"] = int(run)
    pop.initC["weights_sum"] = weights_sum
    pop.initC["full_weights_sum"] = full_weights_sum # metallicity weights * stroop weights
    pop.initC["dco_weights_sum"] = dco_weights_sum # probably redundant now

    pop.final_bpp["x_no_kick"] = x_no_kick
    pop.final_bpp["y_no_kick"] = y_no_kick
    pop.final_bpp["z_no_kick"] = z_no_kick
    pop.final_bpp["v_x"] = finalvx
    pop.final_bpp["v_y"] = finalvy
    pop.final_bpp["v_z"] = finalvz
    pop.final_bpp["component"] = component


    pop.initC["snr10"] = full_snr10
    pop.final_bpp["snr10"] = full_snr10
    pop.initC["snr4"] = full_snr4
    pop.final_bpp["snr4"] = full_snr4
    


    pop.initC.loc[:, "bin_num"] = np.arange(len(pop.initC))
    pop.final_bpp.loc[:, "bin_num"] = np.arange(len(pop.final_bpp))
    pop.kick_info.loc[:, "bin_num"] = np.repeat(np.arange(len(pop.final_bpp)), 2)


    pop.initC.reset_index(drop=True, inplace=True)
    pop.final_bpp.reset_index(drop=True, inplace=True)
    pop.kick_info.reset_index(drop=True, inplace=True)


    if run > 1:

        # adjust based on number in file already
        with pd.HDFStore(filepath, 'r') as store:
            existing_sources = store.get_storer('initC').nrows
        print(f"Existing sources in file: {existing_sources}")

        pop.initC.index += existing_sources
        pop.final_bpp.index += existing_sources
        pop.kick_info.index += existing_sources

        pop.initC["bin_num"] += existing_sources
        pop.final_bpp["bin_num"] += existing_sources
        pop.kick_info["bin_num"] += existing_sources

    # Save the run number in the file attributes
    with h5.File(filepath, 'a') as f:
        f.attrs["run_number"] = run
    
    print("Updated run number to ", run)

    # save data
    # debugging save_initC function
    # if run > 1:
    #     uniques = pop.initC.nunique(axis=0)
    #     compress_cols = [col for col in pop.initC.columns if uniques[col] == 1]

    #     trimmed_initC = pop.initC.drop(columns=compress_cols)

    #     with pd.HDFStore(filepath, mode='r') as store:
    #         existing_cols = store['initC'].columns

    #     differ_cols = trimmed_initC.columns.symmetric_difference(existing_cols)
    #     print("Differing columns :", differ_cols)
    #     for col in differ_cols:
    #         print(col, np.unique(trimmed_initC[col]))
    #         with pd.HDFStore(filepath, mode='r') as store:
    #             if col in store['initC'].columns:
    #                 existing_uniques = np.unique(store['initC'][col])
    #                 print(f"Existing uniques for {col}: {existing_uniques}")
    #             else:
    #                 existing_uniques = np.unique(store['initC_settings'][col])
    #                 print(f"Existing uniques for {col} in settings: {existing_uniques}")

    save_initC(filepath, pop.initC, key='initC', settings_key='initC_settings', force_save_all=False)
    pop.final_bpp.to_hdf(filepath, key='final_bpp', mode='a', format='table', append=True)
    pop.kick_info.to_hdf(filepath, key='kick_info', mode='a', format='table', append=True)

    # print(f"Number of sources with SNR > 7: {len(LISA_sources)}")
    print(f"Saved sources to file for {btype} run {run}")
    print(f"Time taken for run {run}: {time.time() - time_start:.2f} seconds")
