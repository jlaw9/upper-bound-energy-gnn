import argparse
import gzip
import json
import re
import os
from pathlib import Path
from distutils.log import warn

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from tqdm.auto import tqdm

tqdm.pandas()


class AtomicNumberPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num


def preprocess_structure(row):
    inputs = preprocessor(row.structure, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    if np.isclose(min_distance, 0):
        warn(f"Error with {row.id}")
        return None

    scale_factor = 1.0 / inputs["distance"].min()
    inputs["distance"] *= scale_factor

    return pd.Series({
        'inputs': inputs,
        'scale_factor': scale_factor,
    })


def get_structures(structures_file):
    """ Load and preprocess structures from a pymatgen json.gz file
    """
    print(f"Reading {structures_file}")
    with gzip.open(structures_file, "r") as f:
        for key, structure_dict in tqdm(json.loads(f.read().decode()).items()):
            structure_dict = json.loads(structure_dict)
            structure = Structure.from_dict(structure_dict)
            yield {"id": key, "structure": structure}



preprocessor = AtomicNumberPreprocessor()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=Path,
                        nargs=3, action='append',
                        metavar=('structures_file', 'energy_file', 'dataset_name'),
                        help="Specify a structures .json.gz file, "
                             "a CSV with the energies of the structures (at least an 'id' and 'energyperatom' column), "
                             "and a name for the dataset e.g., relaxed")
    parser.add_argument('--out-file', type=Path, default="inputs/preprocessed/scaled_inputs.p",
                        help='path/to/output pickle file containing a pandas dataframe of the preprocessed structures')
    args = parser.parse_args()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

#    structure_dir = Path("/projects/rlmolecule/jlaw/inputs/structures")
#    inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
#    volrelax_dir = Path("/projects/rlmolecule/pstjohn/volume_relaxation_outputs/")
    all_data = pd.DataFrame()
    for structures_file, energies_file, name in args.dataset:
        structures = pd.DataFrame(get_structures(structures_file))
        print(f"\t{len(structures)} structures read")

        print(f"Reading {energies_file}")
        energies = pd.read_csv(energies_file, index_col='id')
        energies['dataset'] = str(name)
        print(f"\t{len(energies)} structures")

        energies['structure'] = structures.set_index('id').structure
        energies = energies.dropna(subset=['structure'])
        print(f"{len(energies)} structures after merging")
        print(energies.head(2))

        all_data = pd.concat([all_data, energies])

    all_data = all_data.reset_index()
    #data = pd.read_pickle(Path(inputs_dir, "20220603_all_structures.p"))
    preprocessed = all_data.progress_apply(preprocess_structure, axis=1)
    data = all_data.join(preprocessed, how='inner')
    data = data.dropna(subset=['inputs']).drop(["structure"], axis=1)

    print(f"Writing {len(data)} processed structures to {args.out_file}")
    data.to_pickle(args.out_file)
