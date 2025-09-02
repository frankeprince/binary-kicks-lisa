import cogsworth
import numpy as np
import pandas as pd
import astropy.units as u
from schwimmbad import MultiPool

import h5py as h5


class StroopPop(cogsworth.pop.Population):
    def __init__(self, stroop_sample, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stroop_sample = stroop_sample
        
    def sample_initial_binaries(self):
        """
        Sample initial binaries from the Stroop sample.
        """
        # Calculate galaxy weights
        # grid = np.linspace(-4, np.log10(0.03), 50)
        # inner_bins = np.array([grid[i] + (grid[i+1] - grid[i]) / 2 for i in range(len(grid) - 1)])
        # bins = 10**np.hstack((grid[0], inner_bins, grid[-1]))
        # bin_centers = 10**np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins) - 1)])
        # g = self.sfh_model(size=10_000_000, **self.sfh_params)
        # counts, _ = np.histogram(g.Z.value, bins=bins)
        # self.MW_weights = counts / np.sum(counts)

        # Add weights to Stroop sample
        # stroop_bins = np.digitize(self.stroop_sample.metallicity.values, bins) - 1
        # mapped_weights = np.take(self.MW_weights, stroop_bins)
        # self.stroop_sample["MW_weights"] = mapped_weights

        # Sample binaries
#         self._initial_binaries = self.stroop_sample.sample(
#             n=self.n_binaries_match,
#             weights=self.stroop_sample["MW_weights"].values * self.stroop_sample["mixture_weight"].values,
#             replace=True)
        self._initial_binaries = self.stroop_sample

        # sort by metallicity
        order = np.argsort(self._initial_binaries.metallicity.values)
        self._initial_binaries = self._initial_binaries.iloc[order]

        # reset index
        self._initial_binaries.reset_index(drop=True, inplace=True)
        
        # temp fix: Reset bin_nums, store old bin_nums in new column
        self._initial_binaries["old_bin_num"] = self._initial_binaries["bin_num"]
        self._initial_binaries["bin_num"] = np.arange(self.n_binaries_match)

        self.sample_initial_galaxy()

        # update the metallicity and birth times of the binaries to match the galaxy
        # self._initial_binaries["metallicity"] = self._initial_galaxy.Z
        self._initial_binaries["tphysf"] = self._initial_galaxy.tau.to(u.Myr).value

    def sample_initial_galaxy(self):
        """
        Sample a galaxy to roughly match metallicity of binaries.
        """
        self._initial_galaxy = self.sfh_model(size=0, **self.sfh_params)

        # Get bins, counts, from initial binaries
        maxZ = np.max(self._initial_binaries.metallicity.values) * 1.01
        minZ = np.min(self._initial_binaries.metallicity.values) * 0.99
        grid = np.linspace(np.log10(minZ), np.log10(maxZ), 50)
        inner_bins = np.array([grid[i] + (grid[i+1] - grid[i]) / 2 for i in range(len(grid) - 1)])
        self.bins = 10**np.hstack((grid[0], inner_bins, grid[-1]))
        self.bin_centers = 10**np.array([(self.bins[i] + self.bins[i+1])/2 for i in range(len(self.bins) - 1)])

        counts, _ = np.histogram(self._initial_binaries.metallicity.values, bins=self.bins)
        
        divided_counts = np.floor_divide(counts, self.processes * np.ones(50, dtype=int))
        remainder_counts = counts - divided_counts * self.processes
        counts_list = [divided_counts.copy() for _ in range(self.processes)]
        counts_list[0] += remainder_counts  # Add the remainder to the first process

        with MultiPool() as pool:
            galaxies = list(pool.map(self.fill_galaxy_bins, counts_list))
        print(galaxies)
        
        # Concatenate galaxies from all processes
        self._initial_galaxy = cogsworth.sfh.concat(*galaxies)
        
        # Initialize galaxy counts ### No Multiprocessing
#         galaxy_counts = np.zeros_like(counts)
#         while np.sum(galaxy_counts) < np.sum(counts):
#             # print(self._initial_galaxy)
#             # sample galaxy locations

#             g = self.sfh_model(size=100_000_000, **self.sfh_params)

#             for i in range(len(counts)):
#                 if galaxy_counts[i] == counts[i]:
#                     continue
#                 i_locations = g[(self.bins[i] <= g.Z.value) & (g.Z.value < self.bins[i+1])]
#                 if (galaxy_counts[i] + len(i_locations) > counts[i]):
#                     random_inds = np.random.choice(len(i_locations), size=counts[i]-galaxy_counts[i], replace=False)
# #                     self._initial_galaxy = cogsworth.sfh.concat(self._initial_galaxy, 
# #                                                                 i_locations[:int(counts[i] - galaxy_counts[i])])
#                     self._initial_galaxy = cogsworth.sfh.concat(self._initial_galaxy, 
#                                                                 i_locations[random_inds])
#                     galaxy_counts[i] += counts[i] - galaxy_counts[i]
#                 else:
#                     galaxy_counts[i] += len(i_locations)
#                     self._initial_galaxy = cogsworth.sfh.concat(self._initial_galaxy, i_locations)
        # sort by metallicity
        # print(self._initial_galaxy)
        order = np.argsort(self._initial_galaxy.Z.value)
        self._initial_galaxy = self._initial_galaxy[order]

        # add relevant citations
        self.__citations__.extend([c for c in self._initial_galaxy.__citations__ if c != "cogsworth"])

        # if velocities are already set then just immediately return
        if all(hasattr(self._initial_galaxy, attr) for attr in ["v_R", "v_T", "v_z"]):   # pragma: no cover
            return

        # work out the initial velocities of each binary
        vel_units = u.km / u.s

        # calculate the Galactic circular velocity at the initial positions
        v_circ = self.galactic_potential.circular_velocity(q=[self._initial_galaxy.x,
                                                              self._initial_galaxy.y,
                                                              self._initial_galaxy.z]).to(vel_units)

        # add some velocity dispersion
        v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                         self.v_dispersion.to(vel_units) / np.sqrt(3),
                                         size=(3, self.n_binaries_match))
        v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units
        self._initial_galaxy.v_R = v_R
        self._initial_galaxy.v_T = v_T
        self._initial_galaxy.v_z = v_z

    def fill_galaxy_bins(self, counts):
        galaxy_counts = np.zeros_like(counts)
        galaxy = self.sfh_model(size=0, **self.sfh_params)
        while np.sum(galaxy_counts) < np.sum(counts):
            # print(self._initial_galaxy)
            # sample galaxy locations

            g = self.sfh_model(size=100_000, **self.sfh_params)

            # adjust metallicity into cosmic bounds
            g.Z[g.Z < 1e-4] = 1e-4
            g.Z[g.Z > 0.03] = 0.03

            for i in range(len(counts)):
                if galaxy_counts[i] == counts[i]:
                    continue
                i_locations = g[(self.bins[i] <= g.Z.value) & (g.Z.value < self.bins[i+1])]
                if (galaxy_counts[i] + len(i_locations) > counts[i]):
                    random_inds = np.random.choice(len(i_locations), size=counts[i]-galaxy_counts[i], replace=False)
#                     self._initial_galaxy = cogsworth.sfh.concat(self._initial_galaxy, 
#                                                                 i_locations[:int(counts[i] - galaxy_counts[i])])
                    galaxy = cogsworth.sfh.concat(galaxy, 
                                                                i_locations[random_inds])
                    galaxy_counts[i] += counts[i] - galaxy_counts[i]
                else:
                    galaxy_counts[i] += len(i_locations)
                    galaxy = cogsworth.sfh.concat(galaxy, i_locations)
        return galaxy

    def __getitem__(self, ind):
        # convert any Pandas Series to numpy arrays
        ind = ind.values if isinstance(ind, pd.Series) else ind

        # if the population is associated with a file, make sure it's entirely loaded before slicing
        if self._file is not None:
            parts = ["initial_binaries", "bpp", "initial_galaxy", "orbits"]
            vars = [self._initial_binaries, self._bpp, self._initial_galaxy, self._orbits]
            masks = {f"has_{p}": False for p in parts}
            with h5.File(self._file, "r") as f:
                for p in parts:
                    masks[f"has_{p}"] = p in f
            missing_parts = [p for i, p in enumerate(parts) if (masks[f"has_{p}"] and vars[i] is None)]

            if len(missing_parts) > 0:
                raise ValueError(("This population was loaded from a file but you haven't loaded all parts "
                                  "yet. You need to do this before indexing it. The missing parts are: "
                                  f"{missing_parts}. You either need to access each of these variables or "
                                  "reload the entire population using all parts."))

        # ensure indexing with the right type
        ALLOWED_TYPES = (int, slice, list, np.ndarray, tuple)
        if not isinstance(ind, ALLOWED_TYPES):
            raise ValueError((f"Can only index using one of {[at.__name__ for at in ALLOWED_TYPES]}, "
                              f"you supplied a '{type(ind).__name__}'"))

        # check validity of indices for array-like types
        if isinstance(ind, (list, tuple, np.ndarray)):
            # check every element is a boolean (if so, convert to bin_nums after asserting length sensible)
            if all(isinstance(x, (bool, np.bool_)) for x in ind):
                assert len(ind) == len(self.bin_nums), "Boolean mask must be same length as the population"
                ind = self.bin_nums[ind]
            # otherwise ensure all elements are integers
            else:
                assert all(isinstance(x, (int, np.integer)) for x in ind), \
                    "Can only index using integers or a boolean mask"
                if len(np.unique(ind)) < len(ind):
                    warnings.warn(("You have supplied duplicate indices, this will invalidate the "
                                   "normalisation of the Population (e.g. mass_binaries will be wrong)"))

        # set up the bin_nums we are selecting
        bin_nums = ind

        # turn ints into arrays and convert slices to exact bin_nums
        if isinstance(ind, int):
            bin_nums = [ind]
        elif isinstance(ind, slice):
            bin_nums = self.bin_nums[ind]
        bin_nums = np.asarray(bin_nums)

        # check that the bin_nums are all valid
        check_nums = np.isin(bin_nums, self.bin_nums)
        if not check_nums.all():
            raise ValueError(("The index that you supplied includes a `bin_num` that does not exist. "
                              f"The first bin_num I couldn't find was {bin_nums[~check_nums][0]}"))

        # start a new population with the same parameters
        new_pop = self.__class__(stroop_sample=self.stroop_sample, n_binaries=len(bin_nums), processes=self.processes,
                                 m1_cutoff=self.m1_cutoff, final_kstar1=self.final_kstar1,
                                 final_kstar2=self.final_kstar2, sfh_model=self.sfh_model,
                                 sfh_params=self.sfh_params, galactic_potential=self.galactic_potential,
                                 v_dispersion=self.v_dispersion, max_ev_time=self.max_ev_time,
                                 timestep_size=self.timestep_size, BSE_settings=self.BSE_settings,
                                 sampling_params=self.sampling_params,
                                 store_entire_orbits=self.store_entire_orbits)
        new_pop.n_binaries_match = new_pop.n_binaries

        # proxy for checking whether sampling has been done
        if self._mass_binaries is not None:
            new_pop._mass_binaries = self._mass_binaries
            new_pop._mass_singles = self._mass_singles
            new_pop._n_singles_req = self._n_singles_req
            new_pop._n_bin_req = self._n_bin_req

        bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums)}
        sort_idx = np.argsort(list(bin_num_to_ind.keys()))
        idx = np.searchsorted(list(bin_num_to_ind.keys()), bin_nums, sorter=sort_idx)
        inds = np.asarray(list(bin_num_to_ind.values()))[sort_idx][idx]

        if self._initial_galaxy is not None:
            new_pop._initial_galaxy = self._initial_galaxy[inds]
        if self._initC is not None:
            new_pop._initC = self._initC.loc[bin_nums]
        if self._initial_binaries is not None:
            new_pop._initial_binaries = self._initial_binaries.loc[bin_nums]

        # checking whether stellar evolution has been done
        if self._bpp is not None:
            # copy over subsets of data when they aren't None
            new_pop._bpp = self._bpp.loc[bin_nums]
            if self._bcm is not None:
                new_pop._bcm = self._bcm.loc[bin_nums]
            if self._kick_info is not None:
                new_pop._kick_info = self._kick_info.loc[bin_nums]
            if self._final_bpp is not None:
                new_pop._final_bpp = self._final_bpp.loc[bin_nums]
            if self._disrupted is not None:
                new_pop._disrupted = self._disrupted[inds]
            if self._classes is not None:
                new_pop._classes = self._classes.iloc[inds]
            if self._observables is not None:
                new_pop._observables = self._observables.iloc[inds]

            if self._orbits is not None or self._final_pos is not None or self._final_vel is not None:
                disrupted_bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums[self.disrupted])}
                sort_idx = np.argsort(list(disrupted_bin_num_to_ind.keys()))
                idx = np.searchsorted(list(disrupted_bin_num_to_ind.keys()),
                                      bin_nums[np.isin(bin_nums, self.bin_nums[self.disrupted])],
                                      sorter=sort_idx)
                inds_with_disruptions = np.asarray(list(disrupted_bin_num_to_ind.values()))[sort_idx][idx]\
                      + len(self)
                all_inds = np.concatenate((inds, inds_with_disruptions)).astype(int)

            # same thing but for arrays with appended disrupted secondaries
            if self._orbits is not None:
                new_pop._orbits = self.orbits[all_inds]
            if self._final_pos is not None:
                new_pop._final_pos = self._final_pos[all_inds]
            if self._final_vel is not None:
                new_pop._final_vel = self._final_vel[all_inds]
        return new_pop
