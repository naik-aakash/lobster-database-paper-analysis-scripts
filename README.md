This repository consists of analysis scripts to reproduce the publication on LOBSTER database

[![DOI](https://zenodo.org/badge/606380090.svg)](https://zenodo.org/badge/latestdoi/606380090)

## Workflow initialization

The following version numbers are used for the workflows:
- [pymatgen 2022.11.7](https://pypi.org/project/pymatgen/2022.11.7/)
- [atomate 1.0.3](https://pypi.org/project/atomate/1.0.3/)
- [custodian 2023.3.10](https://pypi.org/project/custodian/2023.3.10/)

`Workflow.ipynb` includes script to start all LOBSTER computations with
pymatgen, fireworks, and atomate.

## Data generation and reuse
- Use the `Data_generation/requirements.txt` to create conda environment with necessary packages
- The `Lobster_lightweight_json_generation.ipynb` script will generate light weight lobster jsons that consists of lobsterpy summarzied bonding information, relevant strongest bonds, madelung energies of the structures and atomic charges (refer Table 1 and 2 of the manuscript for the description). 
- The `Computational_data_generation.ipynb` script stores all the relevant LOBSTER computation files in the form of JSON using pydantic schema as implemented for atomate2 (refer Table 3 of the manuscript for the description). 

  ### Reuse / Read data records
  - `Example_data/Lightweight_jsons/` -- path to sample LOBSTER Lightweight JSONS files
  - `Example_data/Computational_data_jsons/` -- path to sample Computational JSON files
  - All 1520 LOBSTER Lightweight JSONS / Computational data JSONS can be download here : 
    - [10.5281/zenodo.7794812](https://doi.org/10.5281/zenodo.7794812) 
    - [10.5281/zenodo.7821728](https://doi.org/10.5281/zenodo.7821728)
  - Use the `Read_data_records/requirements.txt` to create conda environment with necessary packages
  - `Read_lobsterpy_data.ipynb` This script will read LobsterPy summarized bonding information JSON files as python dictionary (refer Table 1 of the manuscript for the description). 
  - `Read_lobsterschema_data.ipynb` This script will read LobsterSchema data as pymatgen objects and consists of all the relevant LOBSTER computation data in the form of python dictionary (refer Table 2 of the manuscript for the description).

## Validating the results
- [LobsterPy 0.2.9](https://pypi.org/project/lobsterpy/0.2.9/)
- [atomate2](https://github.com/materialsproject/atomate2/tree/fa603e3cb4c3024b9b12b0d752793a9191d99f8a) - Install it using `pip install git+https://github.com/materialsproject/atomate2/tree/fa603e3cb4c3024b9b12b0d752793a9191d99f8a`
- [dash 2.8.1](https://pypi.org/project/dash/2.8.1/)
- [seaborn 0.12.2](https://pypi.org/project/seaborn/0.12.2/)
- [plotly 5.10.0](https://pypi.org/project/plotly/5.10.0/)
- [pymatviz 0.5.1](https://pypi.org/project/pymatviz/0.5.1/)

  ### Technical validation
  - Download all the computational data files from following repository links:
    - [Part 1](https://doi.org/10.5281/zenodo.7852083), [Part 2](https://doi.org/10.5281/zenodo.7852108), [Part 3](https://doi.org/10.5281/zenodo.7852792), [Part 4](https://doi.org/10.5281/zenodo.7852799), [Part 5](https://doi.org/10.5281/zenodo.7852807), [Part 6](https://doi.org/10.5281/zenodo.7852809), [Part 7](https://doi.org/10.5281/zenodo.7852821), [Part 8](https://doi.org/10.5281/zenodo.7852824)
    - Make a directory named `Results` to extract all the tar files.
    - For example , extract `Part1.tar file`, using following command `tar -xf Part1.tar -C ./Results/`
    - Repeat the command above to extract all the 8 tar files to `Results` directory.
    - This should result in 1520 directories inside `Results`. Each sub directory will be named as `mp-xxx` and it denotes the Materials Project ID of the compound.
  - You can then use the scripts provided to reproduce our technical validation section results.
  
  ### Charge spilling

  - `Charge_spilling_lobster.ipynb` will produce the dataframe with charge spillings for entire dataset and also create the histograms (as in the manuscript). 
  - `Charge_spilling_data.pkl` consists of presaved data from `Charge_spilling_lobster.ipynb` script run (load this to get plots on the go).
  
  ### Band overlaps

  - `Band_overlaps.ipynb` will produce the dataframe with devaitons from `bandOverlaps.lobster` for entire dataset. 
  - `Band_overlaps_data.pkl` consists of presaved data from `Band_overlaps.ipynb` script run (load this to get results on the go).

  ### DOS comparisons
  - `Get_plots_band_features_tanimoto.ipynb` will produce all the PDOS benchmarking data, save pandas dataframes as pickle and also save the all the plots
  - `lsolobdos.pkl` and `lobdos.pkl` consists of all the data necessary to reproduce the plots (as shown in Fig 4, SI Fig S1, S2, S3) 
  - `Save_pdos_plot_and_data.ipynb` will save the PDOS comparison plots.
  - ##### Interactive visualization of PDOS benchmark plots 
    1. Download the dash app and its data from  [10.5281/zenodo.7795903](https://zenodo.org/record/7795903#.ZCv1yXvP1PY)
    2. Run the `Band_features.py` script to get dash app to explore all the s, p, d band feature plots (Checkout -h options)
    3. Run the `Check_fingerprints.py` script to get dash app to visualize all the s, p, d fingerprint plots (Checkout -h options)s

  ### Charge and coordination comp
  - `BVA_Charge_comparisons.ipynb` will produce the results of charge comparison analysis and also corresponding plots (as shown in SI, Fig S4,S5)
  - `Charge_comp_data.pkl` contains saved to charge comparison 
  - `Coordination_comparisons_BVA.ipynb` will produce the results of coordination environments comparisons 
  - `Coordination_comp_data_bva.pkl` contains saved to coordination environments comparisons

  ### Data overview
  - `Data_topology.ipynb` this script will extract and store the data necessary for Fig 5.
  - `Lobster_dataoverview.pkl` contains presaved data ready to be used for generating Fig 5.

  ## ML model
  - Use the `ML_model/requirements.txt` to create conda environment with necessary packages
  - `mpids.csv` File contains list of material project ids and corresponding compositions
  - `featurizer` This python module is used to featurize lobster lightweight jsons to use ICOHP data as features for ML model
  - `Featurize_lobsterpy_jsons.ipynb` This script will generate lobster  features via featurizer module save it as using the featurizer module `lobsterpy_featurized_data.csv`
  - `ML_data_with_automatminer.ipynb` This script uses automatminer featurizer to extract matminer features based on composition and structure and creates data ready to be used for ML model training (also adds lobter summary stats data as features)- `dataforml_automatminer.pkl`
  - `ml_utilities.py` This module contains utility functions used for training and evaluating random forest (RF) regressor models. 
  - `RF_model.ipynb` This script will train and evaluate 2 RF regressor models using nested CV approach. (Including and exclusing LOBSTER features)
  - `Automatminer_rf_ml_model.ipynb` This script will train and evaluate RF regression models using automatminer Matpipe (Used to compare matbench RF model).
  - `exc_icohp` This directory containts model cross validation evaluation result plot and feature importance plots
  - `exc_icohp/summary_stats.csv` This file containts summarized stats of model trained and evaluated using `RF_model.ipynb` script. (Excluding LOBSTER features)
  - `inc_icohp` This directory containts model cross validation evaluation result plot and feature importance plots
  - `inc_icohp/summary_stats.csv` This file containts summarized stats of model trained and evaluated using `RF_model.ipynb` script. (Including LOBSTER features)
  - `Plot_summary_results.ipynb` This scripts reads the `summary_stats.csv` of the RF model and visualizes data from Table 4. 

