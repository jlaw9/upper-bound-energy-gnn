## Structure Datasets
1. `ICSD`: 16,409 ground-state structures. These cannot be distributed.
2. `Fully-relaxed`: 64,958 structures for which a full DFT relaxation was performed. There are a total of 772 compositions among these structures.
3. `Volume-relaxed`: 58,240 structures for which a constrained DFT relaxation was performed, where the atom positions and cell shape were held constant. 
We performed additional quality control filters for these structures.

## Other files
- `competing_phases.csv`: Table of composition - energy pairs from the NREL MatDB against which structures are evaluated for stability.
- `POSCAR_example`: This is the LiSc2F7 relaxed structure from Fig. 6 of the paper.
