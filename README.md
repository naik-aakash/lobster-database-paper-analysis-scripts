This repository consists of analysis scripts to reproduce the publication on LOBSTER database 

## Workflow initialization

The following version numbers are used for the workflows:
- [pymatgen 2022.11.7](https://pypi.org/project/pymatgen/2022.11.7/)
- [atomate 1.0.3](https://pypi.org/project/atomate/1.0.3/)
- [custodian 2023.3.10](https://pypi.org/project/custodian/2023.3.10/)

`Workflow.ipynb` includes script to start all LOBSTER computations with
pymatgen, fireworks, and atomate.

## Analyzing the results & data generation
- [LobsterPy 0.2.5](https://github.com/JaGeo/LobsterPy)
- [atomate2 0.0.10](https://pypi.org/project/atomate2/0.0.10/)
- [dash 2.8.1](https://pypi.org/project/dash/2.8.1/)
- [seaborn 0.12.2](https://pypi.org/project/seaborn/0.12.2/)
- [plotly 5.10.0](https://pypi.org/project/plotly/5.10.0/)

### Data generation
- The `Lobsterpy_jsons.ipynb` script will run LobsterPy and store the results as JSON files (refer Table 1 of the manuscript for the description). 
- The `Lobsterschema_jsons.ipynb` script stores all the relevant LOBSTER computation files in the form of JSON using pydantic schema as implemented for atomate2 (refer Table 2 of the manuscript for the description).

### Technical validation
You need to download the data from www.xxx.com first.
At the moment you can find get the data here: `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/`


You can then use the scripts therein to reproduce our technical validation section results.
### Charge spilling

- `Charge_spilling_lobster.ipynb` will produce the dataframe with charge spillings for entire dataset and also create the histograms (as in the manuscript). 
- `Charge_spilling_data.pkl` consists of presaved data from `Charge_spilling_lobster.ipynb` script run (load this to get plots on the go).

### DOS comparisons
- `Get_plots_band_features_tanimoto.ipynb` will produce all the PDOS benchmarking data, save pandas dataframes as pickle and also save the all the plots
- `lsolobdos.pkl` and `lobdos.pkl` consists of all the data necessary to reproduce the plots (as shown in Fig 4, SI Fig S1, S2, S3) 
- `Save_pdos_plot_and_data.ipynb` will save the PDOS comparison plots.
- ##### Interactive visualization of PDOS benchmark plots 
  1. Navigate to `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/LSODOS_plots/` 
  2. Run the `Band_features.py` script to get dash app to explore all the s, p, d band feature plots (Checkout -h options)
  3. Run the `Check_fingerprints.py` script to get dash app to visualize all the s, p, d fingerprint plots (Checkout -h options)

### Charge and coordination comp
- `BVA_Charge_comparisons.ipynb` will produce the results of charge comparison analysis and also corresponding plots (as shown in SI, Fig S4,S5)
- `Charge_comp_data.pkl` contains saved to charge comparison 
- `Coordination_comparisons_BVA.ipynb` will produce the results of coordination environments comparisons 
- `Coordination_comp_data_bva.pkl` contains saved to coordination environments comparisons
- ##### Bader comparisons scripts (optional)
  - `Retreive_bader_charges_aflowlib.ipynb` will get Bader charge data from AFLOW database by using structure matcher from pymatgen
  - `bader_charges.pkl` consists of retrieved Bader charge data (load this to directly start with comparisons) 
  - `Bader_Charge_comparisons.ipynb` will compare Bader charges with Mulliken and Löwdin charges (cation-anion classification only)
  - `Charge_comp_data_bader.pkl` results of Bader charge comparison data
  - `Coordination_comparisons_bader.ipynb` will compare coordination environments from simplest Chemenv strategy using Bader charges as valences to LOBSTER environments based on ICOHP
  - `Coordination_comp_data_bader.pkl` contains results of coordination environment comparisons using Bader charges as valences

### Data topoplogy
- `Data_topology.ipynb` this script will extract and store the data necessary for Fig 5.
- `Lobster_dataoverview.pkl` contains presaved data ready to be used for generating Fig 5.

### Read data records
- `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/Lobsterpy_json/` -- path to summarized bonding data files
- `/hpc-user/AG-JGeorge/anaik/Phonon_dataset_LSO/Json_data/` -- path to LobsterSchema data files
- `Read_lobsterpy_data.ipynb` This script will read LobsterPy summarized bonding information JSON files as python dictionary (refer Table 1 of the manuscript for the description). 
- `Read_lobsterschema_data.ipynb` This script will read LobsterSchema data as pymatgen objects and consists of all the relevant LOBSTER computation data in the form of python dictionary (refer Table 2 of the manuscript for the description).

### ML model
- `Featurize_lobsterpy_jsons.ipynb` This script generates summary stats data from lobsterpy json files and save it as `Small_basis_summary_stat.csv` (Still needs to be updated)
- `matbench_data_with_mpid.pkl` This file contains target property for predictions (last phdos peak) with structure and corresponding materials project id (material ids are added using structure matcher)
- `Modnet_featurizer.ipynb` This script uses modnet featurizer to extract matminer features based on composition and structure and creates data ready to be used for ML model training - `dataforml.pkl`
- `All_models.ipynb` This script will train and evaluate 6 different regression models using nested CV approach.
