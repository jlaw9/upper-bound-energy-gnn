
import logging
from typing import Dict, List, Optional, Tuple
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from src import reward_utils
from src import ehull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructureRewardBattInterface:
    """ Compute the reward for crystal structures
    """

    def __init__(self,
                 competing_phases: List[PDEntry],
                 reward_weights: dict = None,
                 reward_cutoffs: dict = None,
                 cutoff_penalty: float = 2,
                 sub_rewards: Optional[List[str]] = None,
                 **kwargs) -> None:
        """ A class to estimate the suitability of a crystal structure as a solid state battery interface.

        :param competing_phases: list of competing phases used to 
            construct the convex hull for the elements of the given composition
        :param reward_weights: Weights specifying how the individual rewards will be combined.
        For example: `{"decomp_energy": 0.5, "cond_ion_frac": 0.1, [...]}`
        :param sub_rewards: Use only the specified sub rewards e.g., ['decomp_energy']
        :param cutoff_penalty: If the sub-reward does not pass the cutoff, then apply a penalty.
            The penalty is: scaled_reward / cutoff_penalty
        """
        self.competing_phases = competing_phases
        self.reward_weights = reward_weights
        self.reward_cutoffs = reward_cutoffs
        self.sub_rewards = sub_rewards
        self.cutoff_penalty = cutoff_penalty
        # if the structure passes all cutoffs, then it gets a bonus
        # added to the combined reward
        self.cutoff_bonus = .25
        # For these rewards, smaller is better
        self.rewards_to_minimize = ['decomp_energy', 'oxidation']

        # set the weights of the individual rewards
        if self.reward_weights is None:
            self.reward_weights = {"decomp_energy": 2/3,
                                   "cond_ion_frac": 1/6,
                                   #"cond_ion_vol_frac": .1,
                                   "reduction": 1/18,
                                   "oxidation": 1/18,
                                   "stability_window": 1/18,
                                   }
        # set the cutoffs for the individual rewards
        # If the value does not fall in the desired range,
        # then apply a penalty.
        # The penalty is: scaled_reward / cutoff_penalty
        if self.reward_cutoffs is None:
            self.reward_cutoffs = {"decomp_energy": -0.1,
                                   "cond_ion_frac": .3,
                                   #"cond_ion_vol_frac": .3,
                                   "reduction": -2,
                                   "oxidation": -4,
                                   "stability_window": 2,
                                   }
        self.reward_ranges = {"decomp_energy": (-1, 5),
                              "cond_ion_frac": (0, 0.6),
                              #"cond_ion_vol_frac": (0, 0.8),
                              "reduction": (-5, 0),
                              "oxidation": (-5, 0),
                              "stability_window": (0, 5),
                              }
        self.default_decomp_energy = self.reward_ranges['decomp_energy'][1]

        # make sure the different reward dictionaries line up
        matching_keys = (set(self.reward_weights.keys())
                         & set(self.reward_cutoffs.keys())
                         & set(self.reward_ranges.keys()))
        assert len(matching_keys) == len(self.reward_weights), \
               (f"reward_weights (len = {len(self.reward_weights)}), "
                f"reward_cutoffs (len = {len(self.reward_cutoffs)}), "
                f"and reward_ranges (len = {len(self.reward_ranges)}), "
                f"must have matching keys. Keys that match all three: {matching_keys}")

        if self.sub_rewards is not None:
            num_matching = len(set(self.sub_rewards)
                               & set(self.reward_weights.keys()))
            assert num_matching == len(self.sub_rewards), \
                   "sub_rewards must be a subset of reward_weights"
            # don't use a cutoff bonus if not all subrewards are being used
            if len(self.sub_rewards) < len(self.reward_weights):
                print(f"Using {len(self.sub_rewards)} rewards: "
                      f"{self.sub_rewards}")
                print("Setting cutoff_bonus to 0")
                self.cutoff_bonus = 0

    def compute_reward(self,
                       composition: Composition,
                       predicted_energy: float = None,
                       strc_id: str = None,
                       ):
        """
        The following sub-rewards are combined:
        1. Decomposition energy: predicts the total energy using a GNN model
            and calculates the corresponding decomposition energy based on the competing phases.
        2. Conducting ion fraction
        3. Conducting ion volume
        4. Reduction potential
        5. Oxidation potential
        6. Electrochemical stability window:
            difference between 4. and 5.
        
        Returns:
            float: reward
            dict: info
        """
        sub_rewards = {}
        info = {}
        if predicted_energy is None:
            decomp_energy = self.default_decomp_energy
        else:
            info.update({'predicted_energy': predicted_energy})
            decomp_energy, stability_window = ehull.convex_hull_stability(
                    composition,
                    predicted_energy,
                    self.competing_phases,
            )
            if decomp_energy is None:
                # subtract 1 to the default energy to distinguish between
                # failed calculation here, and failing to decorate the structure 
                decomp_energy = self.default_decomp_energy - 1
            else:
                if decomp_energy < 0 and stability_window is not None:
                    oxidation, reduction = stability_window
                    stability_window_size = reduction - oxidation
                    sub_rewards.update({'oxidation': oxidation,
                                        'reduction': reduction,
                                        'stability_window': stability_window_size})

        sub_rewards['decomp_energy'] = decomp_energy

        try:
            cond_ion_frac = reward_utils.get_conducting_ion_fraction(composition)
            sub_rewards['cond_ion_frac'] = cond_ion_frac

            #cond_ion_vol_frac = reward_utils.compute_cond_ion_vol(structure)
            #sub_rewards['cond_ion_vol_frac'] = cond_ion_vol_frac

        # some structures don't have a conducting ion
        except ValueError as e:
            print(f"ValueError: {e}. ID: {strc_id}")

        info.update({s: round(r, 4) for s, r in sub_rewards.items()})
        if self.sub_rewards is not None:
            # limit to the specified sub rewards
            sub_rewards = {n: r for n, r in sub_rewards.items() if n in self.sub_rewards}
        combined_reward = self.combine_rewards(sub_rewards)
        #print(str(state), combined_reward, info)

        return combined_reward, info

    # This actually takes longer than just computing the convex hull each time
#    def precompute_convex_hulls(self, compositions):
#        self.phase_diagrams = {}
#        for comp in tqdm(compositions):
#            comp = Composition(comp)
#            elements = set(comp.elements)
#            curr_entries = [e for e in self.competing_phases
#                            if len(set(e.composition.elements) - elements) == 0]
#
#            phase_diagram = PhaseDiagram(curr_entries, elements=elements)
#            self.phase_diagrams[comp] = phase_diagram

    def combine_rewards(self, raw_scores, return_weighted=False) -> float:
        """ Take the weighted average of the normalized sub-rewards
        For example, decomposition energy: 1.2, conducting ion frac: 0.1.
        
        :param raw_scores: Dictionary with the raw scores
            for each type of sub-reward e.g., 'decomp_energy': 0.2
        :param return_weighted: Return the weighted sub-rewards 
            instead of their combined sum
        """
        scaled_rewards = {}
        passed_cutoffs = True
        for key, score in raw_scores.items():
            if score is None:
                continue

            reward = score
            # flip the score if lower is better,
            # so that a higher reward is better
            if key in self.rewards_to_minimize:
                reward = -1 * score

            # get the reward bounds
            r_min, r_max = self.reward_ranges[key]
            # also flip the ranges if necessary
            if key in self.rewards_to_minimize:
                r_max2 = -1 * r_min
                r_min = -1 * r_max
                r_max = r_max2

            assert r_max > r_min

            # apply the bounds to make sure the values are in the right range
            reward = max(r_min, reward)
            reward = min(r_max, reward)

            # scale between 0 and 1 using the given range of values
            scaled_reward = (reward - r_min) / (r_max - r_min)

            # If the value does not fall in the desired range,
            # then apply a penalty
            if key in self.reward_cutoffs and len(raw_scores) > 1:
                cutoff = self.reward_cutoffs[key]
                if key in self.rewards_to_minimize:
                    cutoff = -1 * cutoff
                if reward < cutoff:
                    scaled_reward /= float(self.cutoff_penalty)
                    passed_cutoffs = False

            scaled_rewards[key] = scaled_reward

        # Now apply the weights to each sub-reward
        weighted_rewards = {k: v * self.reward_weights[k]
                            for k, v in scaled_rewards.items()}
        combined_reward = sum([v * self.reward_weights[k]
                               for k, v in scaled_rewards.items()])
        # If this structure passed all the cutoffs,
        # then add a bonus to the reward
        if passed_cutoffs and len(raw_scores) > 1:
            combined_reward += self.cutoff_bonus
        if return_weighted:
            return weighted_rewards
        else:
            return combined_reward

