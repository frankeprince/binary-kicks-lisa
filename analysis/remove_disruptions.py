# imports
import pandas as pd
import numpy as np
from StroopPop import StroopPop
import h5py

import astropy.units as u
import argparse

np.random.seed(42)

parser=argparse.ArgumentParser()
# parser.add_argument('--runs', type=int, default=2500,
#                     help='Number of runs to perform for each binary type')
parser.add_argument('--cores', type=int, default=48,
                    help='Number of cores to use for the runs')
# parser.add_argument('--btype', type=str, default='BHBH')
# parser.add_argument('--reset', action='store_true', help='Reset the output file before running the script')
# parser.add_argument('--model', help = 'Physics variation', default='fiducial')
args = parser.parse_args()

CORES = args.cores

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set display width to avoid line breaks
pd.set_option('display.max_rows', 500)  # Show more rows if needed

per_loop = 200000

models = ['fiducial', 'qc_var',  'alpha_var', 'kick_var', 'ecsn_var']

btypes = ['BHBH', 'BHNS', 'BHWD', 'NSNS', 'NSWD']

# for model in models:
for btype in btypes:
    for model in models:
    # for btype in btypes:
        stroopfile = "/hildafs/projects/phy230054p/fep/stroopwafel_dir/stroopwafel/tests/output/{}/{}/{}.h5".format(btype, model, btype)

        print("Processing", stroopfile)
        initC = pd.read_hdf(stroopfile, key='initC')
        print(initC.loc[initC.bin_num == 16468655])
        full_initC = pd.read_hdf(stroopfile, key='full_initC')
        
        binaries = len(initC)
        loops = int(np.ceil(len(initC)/per_loop))

        stroop_nums = []

        for i in range(loops):

            initC = pd.read_hdf(stroopfile, key='initC')[(i+4)*per_loop:min((i+5)*per_loop, binaries)]

            pop = StroopPop(initC, len(initC), processes=CORES, BSE_settings={},
                store_entire_orbits=False)
            pop.BSE_settings = {}
        
            pop.sample_initial_binaries()

            # set all to 13.7 Gyr
            pop._initial_binaries.loc[:, "tphysf"] = (13.7*u.Gyr).to(u.Myr).value

            pop.perform_stellar_evolution()

            disrupted_mask = pop.bpp.sep < 0

            disrupted_nums = pop.bpp.old_bin_num[disrupted_mask].unique()

            # check bin_num 16468655, if it exists
            if 16468655 in pop.initC.old_bin_num.values:
                print("Found 16468655 in this chunk")
                pd.set_option('display.max_columns', None)  # Show all columns
                pd.set_option('display.width', 1000)  # Set display width to avoid line breaks
                pd.set_option('display.max_rows', 500)  # Show more rows if needed
                print(pop.initC.loc[pop.initC.old_bin_num == 16468655])
                print(pop.bpp.loc[pop.bpp.old_bin_num == 16468655, ["tphys", "mass_1", "mass_2", "sep", "kstar_1", "kstar_2"]])
                print("Disrupted?", 16468655 in disrupted_nums)
                quit()

            print(disrupted_nums)
            print(len(disrupted_nums))

            disrupted_initC = pop.initC[pop.initC.old_bin_num.isin(disrupted_nums)]

            # stroop_nums = disrupted_initC.old_bin_num.unique()
            stroop_nums.extend(disrupted_initC.old_bin_num.unique().tolist())

            for disrupted_num in disrupted_nums:
                # check that pop initC and file initC match
                assert np.array_equal(pop.initC.loc[pop.initC.old_bin_num == disrupted_num, "mass_1"].values,
                                       initC.loc[initC.bin_num == disrupted_num, "mass_1"].values)

            # initC = initC.loc[~initC.bin_num.isin(stroop_nums)]
            # initC = initC.reset_index(drop=True)

            # double check that is_hit = 1 for all systems in
            # full_initC.loc[full_initC.bin_num.isin(pop.initC.old_bin_num), 'is_hit'] = 1

            # set is_hit to 0 for all disrupted systems in full_initC
            # full_initC.loc[full_initC.bin_num.isin(stroop_nums), 'is_hit'] = 0


        initC = pd.read_hdf(stroopfile, key='initC')
        full_initC = pd.read_hdf(stroopfile, key='full_initC')

        original_len = len(initC)

        print("Removing", len(stroop_nums), "disrupted systems from", original_len, "initial binaries")

        initC = initC.loc[~initC.bin_num.isin(stroop_nums)]
        initC = initC.reset_index(drop=True)

        assert len(initC) == original_len - len(stroop_nums)

        # double check that is_hit = 1 for all systems in
        full_initC.loc[full_initC.bin_num.isin(initC.bin_num), 'is_hit'] = 1

        # set is_hit to 0 for all disrupted systems in full_initC
        full_initC.loc[full_initC.bin_num.isin(stroop_nums), 'is_hit'] = 0
        
        # delete old keys
        with h5py.File(stroopfile, 'a') as f:
            if 'initC' in f:
                del f['initC']
            if 'full_initC' in f:
                del f['full_initC']
        print("Deleted old data from", stroopfile)
        # resave
        initC.to_hdf(stroopfile, key='initC', mode='a', format='table')
        full_initC.to_hdf(stroopfile, key='full_initC', mode='a', format='table')
        print("Resaved new data to", stroopfile)

        initC = []
        full_initC = []
        pop = []
        quit()


        


