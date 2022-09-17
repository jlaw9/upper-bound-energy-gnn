# upper-bound-energy-gnn

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7089031.svg)](https://doi.org/10.5281/zenodo.7089031)

GNN for predicting an upper bound of the relaxed energy for a given unrelaxed structure

This repo accompanies the paper [Upper-Bound Energy Minimization to Search for Stable Functional Materials with Graph Neural Networks](), _chemRxiv_ 2022.

## Contents
- Pre-trained models: `pretrained_models`
  - The models are available as part of a GitHub Release. Run `bash download.sh` to download them
  - See [model_demonstration.ipynb](https://github.com/jlaw9/upper-bound-energy-gnn/blob/main/model_demonstration.ipynb) for how to load the model and make energy predictions.
- Structure datasets and energies: `inputs`
  - Because of their size, the structures are also part of the GitHub Release.
- Preprocess structures and train a new model: `preprocess.py` and `train_model.py` (see below)
- Top candidate structures and features: `paper_results`

### Data and code organized by figure
- Figure 1: Initial surrogate model development
  c. GNN trained on ICSD and unrelaxed hypothetical structures: `pretrained_models/icsd_and_unrel_battery.hdf5`
- Figure 3: Effect of volume-relaxed dataset on energy and surrogate model performance
  a. Can be recreated with the volume-relaxed and fully-relaxed energies in `inputs`
  b. GNN trained on ICSD, fully-relaxed, and volume-relaxed structures: `pretrained_models/icsd_and_full_vol_battery.hdf5`
    - Code: `train_model.py`
  c. Learning curves.
    - Code: `src/learning_curves`. The submission script may need to be modified for your HPC system
- Figure 4: DFT confirmation of predicted stable structures
  - See `paper_results/dft_confirmation.csv` for the DFT results of 1,707 structures
- Figure 5: Functional features of the predicted stable structures relevant for battery applications
  - See `paper_results/top_candidate_features.csv`
- Figure 6: Crystal structures
  - See `paper_results/relaxed_structures.tar.gz`
- Figure 7: Reinforcement Learning (RL) structure optimization
  - See https://github.com/jlaw9/rl_materials

> Note: To reproduce the results in Figures 4, 5, 6, and 7, use the model `pretrained_models/20220607_icsd_full_and_vol_battery.hdf5`
> This model was trained on a version of the input data without the deduplication filter mentioned in the results

## Installation

Most dependencies for this project are installable via conda, with the exception of [nfp](https://github.com/NREL/nfp),
which can be installed via pip. An example conda environment (yaml) file is provided below:

```yaml
channels:
  - conda-forge
  - defaults
  
dependencies:
  - python=3.7
  - jupyterlab
  - seaborn
  - pandas
  - scikit-learn
  - jupyter
  - notebook
  - pymatgen
  - tqdm
  - tensorflow-gpu
  - pip
    - pip:
    - nfp >= 0.3.12
    - tensorflow-addons
```

### Preprocess structures
To train a model with a different dataset, you first need to prepare them for preprocessing. The `preprocess.py` script expects a csv of the sructure energies with at least the two columns `id` and `energyperatom`, and the pymatgen structures in a `.json.gz` file where the key is the structure ID and the value is the structure. Note the ids between the two files must match.

A `structures.json.gz` file can be created by reading structure files with pymatgen, then using the `structure.as_dict()` function to prepare them to be written to a json file. Here's an example

```python
import gzip, json
from pymatgen.core import Structure

# Read in the structures files
structure = Structure.from_file('inputs/POSCAR_example')
structure_id = '000123'
# Convert them to dictionaries
structures_dict = {structure_id: structure.as_dict()}
# Write to file
with gzip.open('inputs/example.json.gz', 'w') as out:
    out.write(json.dumps(structures_dict, indent=2).encode())
```

As an example for how to run `preprocess.py`, here is the command used to preprocess the three datasets used in the paper: icsd, fully relaxed, and volume-relaxed structures:

```
python preprocess.py \
    --dataset inputs/icsd.json.gz inputs/icsd_energies.csv icsd \
    --dataset inputs/fully_relaxed.json.gz inputs/fully_relaxed_energies.csv relax \
    --dataset inputs/volume_relaxed.json.gz inputs/volume_relaxed_energies.csv vol
```

This will create a pickled pandas dataframe with an 'inputs' column that contains the preprocessed structures.


### Train the GNN model
To train the GNN, use the following command:

```
python train_model.py --inputs inputs/preprocessed/scaled_inputs.p
```

## Cite

```
@article{law2022,
  title={Upper-Bound Energy Minimization to Search for Stable Functional Materials with Graph Neural Networks},
  author={Law, Jeffrey N. and Pandey, Shubham and Gorai, Prashun and John, Peter St},
  journal={chemRxiv},
  year={2022},
  doi = {}
}
```
