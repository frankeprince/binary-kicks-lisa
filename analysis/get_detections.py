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
btypes = ['BHBH', 'BHNS', 'BHWD', 'NSNS', 'NSWD']
btypes = ['BHBH']
k_select_dict = {'NSNS': ([13],[13]), 'BHNS': ([14], [13]), 'BHBH': ([14],[14]), 'BHWD': ([14],[10,11,12]), 'NSWD': ([13], [10, 11, 12]), 'all': ([10, 11, 12, 13, 14], [10, 11, 12, 13, 14]) }



k_select = k_select_dict[btype]
k1_select = k_select[0]
k2_select = k_select[1]

# set filepath
stroopfile = "/hildafs/projects/phy230054p/fep/stroopwafel_dir/stroopwafel/tests/output/{}/{}/{}.h5".format(btype, args.model, btype)
filepath = "../data/LISA_sources/{}/{}/LISA_{}.h5".format(btype, args.model, btype)

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
# print(initC.tphysf)
# for col in initC.columns:
#     print(col)

# find galaxy metallicity weights
grid = np.linspace(-4, np.log10(0.03), 50)
inner_bins = np.array([grid[i] + (grid[i+1] - grid[i]) / 2 for i in range(len(grid) - 1)])
bins = 10**np.hstack((grid[0], inner_bins, grid[-1]))
bin_centers = np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins) - 1)])
g = cogsworth.sfh.Wagg2022(size=10_000_000)
counts, _ = np.histogram(g.Z.value, bins=bins)
MW_weights = counts / np.sum(counts)

# add metallicity weights to initC
initC_bins = np.digitize(initC.metallicity.values, bins) - 1
mapped_weights = np.take(MW_weights, initC_bins)
initC["MW_weights"] = mapped_weights


for run in range(current_run + 1, RUNS + current_run + 1):
    time_start = time.time()

    rand_sample = initC.sample(n=200000, replace=True)
    weights = rand_sample.mixture_weight.values
    metallicity_weights = rand_sample.MW_weights.values
    weights_sum = weights.sum()
    full_weights_sum = (weights * metallicity_weights).sum()
    print(f"Running {btype} run {run} of {RUNS}")
    # Load the population
    pop = StroopPop(n_binaries=len(rand_sample), processes=CORES, BSE_settings={},
            stroop_sample=rand_sample, store_entire_orbits=False)
    
    pop.BSE_settings = {}

    print("Sampling initial binaries")
    sample_start = time.time()
    pop.sample_initial_binaries(reset_sampled_kicks=False)
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
    dco_mask = (pop.bpp["sep"] > 0.0) & full_pairs_mask
    interesting_systems_table = pop.bpp.loc[dco_mask].drop_duplicates(subset = 'bin_num', keep = 'first')
    bin_num_mask = pop.final_bpp.bin_num.isin(interesting_systems_table.bin_num.values)
    dco_weights_sum = np.sum(pop.final_bpp.loc[bin_num_mask].mixture_weight.values)
    print("DCOs formed: ", len(interesting_systems_table))
    print("bpp length: ", len(pop.bpp))
    print("Number of binaries that don't fit pairs: ", len(pop.final_bpp[~pairs_mask]))
    print(f"Time taken to evolve binaries: {time.time() - evol_start:.2f} seconds")
    # throw out binaries that merged or disrupted
    pop = pop[sources_mask & pairs_mask]
    # freq_filter = (1/(pop.final_bpp.porb.values * u.day)).to(u.Hz) > (1e-6*u.Hz)
    # pop = pop[freq_filter]
    if len(pop) > 20000:
        # print("Too many sources, filtering further")
        # ### Source: https://iopscience.iop.org/article/10.3847/2515-5172/ac3d98/ampdf
        # coefficients = np.array([-1.01678, 5.57372, -4.9271, 1.68506])
        # eccentricities = pop.final_bpp.ecc.values
        # frequencies = (1/(pop.final_bpp.porb.values * u.day)).to(u.Hz)
        # harmonic_array = np.ones((len(pop), 4))
        # harmonic_array[:, 0] = eccentricities
        # harmonic_array[:, 1] = eccentricities**2
        # harmonic_array[:, 2] = eccentricities**3
        # harmonic_array[:, 3] = eccentricities**4
        # harmonic_array = np.multiply(harmonic_array, coefficients)
        # harmonic_sum = np.sum(harmonic_array, axis=1)
        # peak_harmonics = 2 * (1 + harmonic_sum) * (1 - eccentricities**2)**(-3/2)
        # peak_frequencies = frequencies * peak_harmonics
        # freq_mask = peak_frequencies > (1e-4 * u.Hz)
        # pop = pop[freq_mask]
        sources = pop.to_legwork_sources(distances=0.1 * u.kpc * np.ones(len(pop)))
        sources.get_snr(t_obs = 10 * u.yr)
        full_snr = sources.snr
        snr_filter = full_snr > 1
        pop = pop[snr_filter]
        print(f"Filtered down to {len(pop)} sources with SNR > 1")
        

    print("Evolving galaxy")

    galaxy_start = time.time()
    pop.perform_galactic_evolution()
    print(f"Time taken to evolve galaxy: {time.time() - galaxy_start:.2f} seconds")

    # Mask for > 10e-4 GW freq
    GWmask = (2/(pop.final_bpp.porb.values * u.day).to(u.s)) > (1e-4*u.Hz)
    var_list = ["mass_1", "mass_2", "porb", "ecc", "metallicity"]
    long_pop = pop.final_bpp[GWmask][var_list].copy()
    print("freq over 1e-4: ", len(long_pop))
    long_x, long_y, long_z = pop.final_pos[GWmask].T.value
    long_pop["x"] = long_x
    long_pop["y"] = long_y
    long_pop["z"] = long_z
    long_pop["run"] = int(run)
    long_pop["weights_sum"] = weights_sum
    long_pop["dco_weights_sum"] = dco_weights_sum



    sources4 = pop.to_legwork_sources(assume_mw_galactocentric=True)
    sources10 = pop.to_legwork_sources(assume_mw_galactocentric=True)
    sources4.get_snr(t_obs = 4 * u.yr)
    sources10.get_snr(t_obs = 10 * u.yr)

    full_snr4 = sources4.snr
    full_snr10 = sources10.snr
    snr10_mask = full_snr10 > 7
    snr4_mask = full_snr4 > 7

    if np.all(~snr10_mask):
        print(f"No sources with SNR > 7 in run {run}. Skipping this run.")
        continue

    LISA_sources = pop[snr10_mask]

    LISA_copy = LISA_sources.copy()

# change to pull position from initial galaxy, t1=0, t2=tau


    v_phi = (LISA_copy.initial_galaxy.v_T / LISA_copy.initial_galaxy.rho)
    v_X = (LISA_copy.initial_galaxy.v_R * np.cos(LISA_copy.initial_galaxy.phi)
               - LISA_copy.initial_galaxy.rho * np.sin(LISA_copy.initial_galaxy.phi) * v_phi)
    v_Y = (LISA_copy.initial_galaxy.v_R * np.sin(LISA_copy.initial_galaxy.phi)
               + LISA_copy.initial_galaxy.rho * np.cos(LISA_copy.initial_galaxy.phi) * v_phi)

    w0s = gd.PhaseSpacePosition(pos=[a.to(u.kpc).value for a in [LISA_copy.initial_galaxy.x,
                                                                     LISA_copy.initial_galaxy.y,
                                                                     LISA_copy.initial_galaxy.z]] * u.kpc,
                                    vel=[a.to(u.km/u.s).value for a in [v_X, v_Y,
                                                                        LISA_copy.initial_galaxy.v_z]] * u.km/u.s)

    no_kick_orbits = [LISA_copy.galactic_potential.integrate_orbit(w0s[i],
                                                       t1=0 * u.Myr,
                                                       t2=LISA_copy.initial_galaxy.tau[i],
                                                       dt=1 * u.Myr)
                  for i in range(len(LISA_copy))]
    x_no_kick = [no_kick_orbits[i][-1].x.value for i in range(len(no_kick_orbits))]
    y_no_kick = [no_kick_orbits[i][-1].y.value for i in range(len(no_kick_orbits))]
    z_no_kick = [no_kick_orbits[i][-1].z.value for i in range(len(no_kick_orbits))]

    snr10 = full_snr10[snr10_mask]
    snr4 = full_snr4[snr10_mask]
    
    pairs_mask = ((LISA_sources.bpp.kstar_1.isin(k1_select)) & (LISA_sources.bpp.kstar_2.isin(k2_select))) | \
                 ((LISA_sources.bpp.kstar_1.isin(k2_select)) & (LISA_sources.bpp.kstar_2.isin(k1_select)))
    
    # add orbits
    LISA_initC = pop.initC.loc[snr10_mask].copy()
    LISA_bin_nums = LISA_initC.bin_num.values
    LISA_formation = LISA_sources.bpp.loc[pairs_mask].drop_duplicates(subset='bin_num').copy()
    # LISA_formation = LISA_formation.loc[LISA_formation.bin_num.isin(LISA_bin_nums)]
    LISA_final = pop.final_bpp.loc[snr10_mask].copy()
    LISA_kicks = pop.kick_info.loc[pop.kick_info.bin_num.isin(LISA_bin_nums)].copy()
    # LISA_orbits = pop.orbits[snr10_mask]

    initialx, initialy, initialz = pop.initial_galaxy.x[snr10_mask].value, pop.initial_galaxy.y[snr10_mask].value, pop.initial_galaxy.z[snr10_mask].value
    finalx, finaly, finalz = pop.final_pos[snr10_mask].T.value
    # initalvx, initalvy, initalvz = pop.initial_galaxy.v_x[snr7_mask], pop.initial_galaxy.v_y[snr7_mask], pop.initial_galaxy.v_z[snr7_mask]
    finalvx, finalvy, finalvz = pop.final_vel[snr10_mask].T.value

    component = pop.initial_galaxy.which_comp[snr10_mask]
    component_map = {
    'low_alpha_disc': 0,
    'high_alpha_disc': 1,
    'bulge': 2}   
    component = np.vectorize(component_map.get)(component)

    LISA_initC["x"] = initialx
    LISA_initC["y"] = initialy
    LISA_initC["z"] = initialz
    LISA_initC["component"] = component

    LISA_final["x"] = finalx
    LISA_final["y"] = finaly
    LISA_final["z"] = finalz
    LISA_final["x_no_kick"] = x_no_kick
    LISA_final["y_no_kick"] = y_no_kick
    LISA_final["z_no_kick"] = z_no_kick
    LISA_final["v_x"] = finalvx
    LISA_final["v_y"] = finalvy
    LISA_final["v_z"] = finalvz
    LISA_final["component"] = component

    LISA_initC["snr10"] = snr10
    LISA_final["snr10"] = snr10
    LISA_formation["snr10"] = snr10
    LISA_initC["snr4"] = snr4
    LISA_final["snr4"] = snr4
    LISA_formation["snr4"] = snr4

    LISA_initC["run"], LISA_formation["run"], LISA_final["run"], LISA_kicks["run"] = run, run, run, run
    LISA_initC["weights_sum"], LISA_formation["weights_sum"], LISA_final["weights_sum"], LISA_kicks["weights_sum"] = weights_sum, weights_sum, weights_sum, weights_sum
    LISA_initC["dco_weights_sum"] = dco_weights_sum
    LISA_formation["dco_weights_sum"] = dco_weights_sum
    LISA_final["dco_weights_sum"] = dco_weights_sum
    LISA_kicks["dco_weights_sum"] = dco_weights_sum

    LISA_formation["stroop_bin_num"] = LISA_initC.stroop_bin_num.values
    LISA_final["stroop_bin_num"] = LISA_initC.stroop_bin_num.values
    LISA_kicks["stroop_bin_num"] = np.repeat(LISA_initC.stroop_bin_num.values, 2)

    LISA_initC["lisa_bin_num"] = np.arange(len(LISA_initC))
    LISA_formation["lisa_bin_num"] = np.arange(len(LISA_formation))
    LISA_final["lisa_bin_num"] = np.arange(len(LISA_final))
    LISA_kicks["lisa_bin_num"] = np.repeat(np.arange(len(LISA_initC)), 2)
    long_pop["lisa_bin_num"] = np.arange(len(long_pop))

    # reset indices/bin_nums
    LISA_initC.reset_index(drop=True, inplace=True)
    LISA_formation.reset_index(drop=True, inplace=True)
    LISA_final.reset_index(drop=True, inplace=True)
    LISA_kicks.reset_index(drop=True, inplace=True)
    long_pop.reset_index(drop=True, inplace=True)


    if run > 1:

        # adjust based on number in file already
        with pd.HDFStore(filepath, 'r') as store:
            existing_sources = store.get_storer('initC').nrows
            high_freq_sources = store.get_storer('high_freq').nrows
        # check orbits count   
        # with h5.File(filepath, 'r') as f:
        #     if 'orbits' in f:
        #         existing_orbits = f['orbits']['offsets'].shape[0] - 1
        print(f"Existing sources in file: {existing_sources}, High freq sources: {high_freq_sources}")
        # print(f"Existing orbits in file: {existing_orbits}")


        LISA_initC.index += existing_sources
        LISA_formation.index += existing_sources
        LISA_final.index += existing_sources
        LISA_kicks.index += existing_sources
        long_pop.index += high_freq_sources

        LISA_initC["lisa_bin_num"] += existing_sources
        LISA_formation["lisa_bin_num"] += existing_sources
        LISA_final["lisa_bin_num"] += existing_sources
        LISA_kicks["lisa_bin_num"] += existing_sources
        long_pop["lisa_bin_num"] += high_freq_sources

    # Save the run number in the file attributes
    with h5.File(filepath, 'a') as f:
        f.attrs["run_number"] = run

    # save data
    long_pop.to_hdf(filepath, key='high_freq', mode='a', format='table', append=True)
    LISA_initC.to_hdf(filepath, key='initC', mode='a', format='table', append=True)
    LISA_formation.to_hdf(filepath, key='formation', mode='a', format='table', append=True)
    LISA_final.to_hdf(filepath, key='final', mode='a', format='table', append=True)
    LISA_kicks.to_hdf(filepath, key='kicks', mode='a', format='table', append=True)

    # save orbits
    # append_orbits_to_file(filepath, LISA_orbits)

    # orbits.append(LISA_orbits)
    # print(orbits)
    # print(LISA_orbits.shape)

    print(f"Number of sources with SNR > 7: {len(LISA_sources)}")
    print(f"Saving sources to file for {btype} run {run+1}")
    print(f"Time taken for run {run}: {time.time() - time_start:.2f} seconds")


    # LISA_sources.save("../data/LISA_sources/{}/{}_run_{}.h5".format(btype, btype, run),
    #               overwrite=True)

    # Add the population to the list
    # pops.append(LISA_sources)
# concatenate the orbits
# orbits = np.concatenate(orbits, axis=0)
# print(orbits)
# save the orbits

# with h5.File(filepath, 'a') as f:
#     f.create_dataset('orbits', data=orbits)
