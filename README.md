This repository consists of analysis scripts to reproduce the publication on Lobster database 

## Workflow initialization

The following version numbers are used for the workflows:
- [pymatgen 2022.11.7](https://pypi.org/project/pymatgen/2022.11.7/)
- [atomate 1.0.3](https://github.com/hackingmaterials/atomate)
- [custodian](https://github.com/materialsproject/custodian) (pip install https://github.com/materialsproject/custodian)

`Workflow.ipynb` includes script to start all Lobster computations with
pymatgen, fireworks, and atomate.

## Analyzing the results & data generation
- [LobsterPy 0.2.5](https://github.com/JaGeo/LobsterPy)
- [atomate2](https://github.com/JaGeo/atomate2/tree/lobster_clean) (pip install git+https://github.com/JaGeo/atomate2.git@lobster_clean)
- dash 2.8.1
- seaborn 0.12.2
- plotly 5.10.0

### Data generation
- `Lobsterpy_jsons.ipynb` This script will run lobsterpy and store the results as jsons (refer Table 1 of the manuscript for the description). 
- `Lobsterschema_jsons.ipynb` This script stores all the relevant LOBSTER computation files in the form of JSON using pydantic schema as implemented for atomate2 (refer Table 2 of the manuscript for the description).

### Technical validation
You need to download the data from www.xxx.com first.
At the moment you can find get the data here: `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/`


You can then use the scripts therein to reproduce our technical validation section results.
### Charge spilling

- `Charge_spilling_lobster.ipynb` will produce the dataframe with charge spillings for entire dataset and also create the histograms (as in the manuscript). 
- `Charge_spilling_data.pkl` consists of presaved data from `Charge_spilling_lobster.ipynb` script run (Load this to get plots on the go) 

### DOS comp
- `Get_plots_band_features_tanimoto.ipynb` will produce all the PDOS benchmarking data, save pandas dataframes as pickle and also save the all the plots
- `lsolobdos.pkl` and `lobdos.pkl` consists of all the data necessary to reproduce the plots (as shown in Fig 4, SI Fig S1, S2, S3) 
- `Save_pdos_plot_and_data.ipynb` will save the PDOS comparison plots.
- ##### Interactive visualization of PDOS benchmark plots 
  1. Navigate to `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/LSODOS_plots/` 
  2. `Band_features.py` run this script to get dash app to explore all the s,p,d band feature plots
  3. `Check_fingerprints.py` run this script to get dash app to visualize all the s,p,d fingerprint plots

### Charge and coordination comp
- `BVA_Charge_comparisons.ipynb` will produce the results of charge comparison analysis and also corresponding plots (as shown in SI, Fig S4,S5)
- `Charge_comp_data.pkl` contains saved to charge comparison 
- `Coordination_comparisons_BVA.ipynb` will produce the results of coordination environments comparisons 
- `Cooridination_comp_data_bva.pkl` contains saved to coordination environments comparisons
- ##### Bader comparisons scripts (optional)
  - `Retreive_bader_charges_aflowlib.ipynb` will get bader charge data from AFLOW database by using structure matcher from pymatgen
  - `bader_charges.pkl` consists of retrieved bader charge data (Load this to directly start with comparisons) 
  - `Bader_Charge_comparisons.ipynb` will compare bader charges with Mulliken and Loewdin (cation-anion classification only)
  - `Charge_comp_data_bader.pkl` results of bader charge comparison data
  - `Coordination_comparisons_bader.ipynb` will compare coordination environments from simplest Chemenv strategy using bader charges as valences to Lobster environments based on ICOHP
  - `Cooridination_comp_data_bader.pkl` contains results of coordination environment comparisons using bader charges as valences

### Data topoplogy
- `Data_topology.ipynb` this script will extract and store the data necessary for Fig 5
- `Lobster_dataoverview.pkl` contains presaved data ready to be used for generating Fig 5.
