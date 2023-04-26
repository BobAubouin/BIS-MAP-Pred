# BIS-MAP-Pred
GitHub associated with the research paper "[Data-based Pharmacodynamic Modeling for BIS and Mean Arterial Pressure Prediction during General Anesthesia](https://hal.science/hal-04066401v1)" accepted for ECC23.

## Structure

    .
    ├─── lee_code # folder to reproduce the method of lee et al.
         ├─── code_lee_v2.py # code from Lee et al. modified for tensorflow v2
         ├─── create_dataset_lee.py # create the dataset that can be used by code_lee_v2.py
         ├─── test_lee.py # ??
    ├─── data # folder to store the data used in the code
         ├─── info_clinic_vitalDB.csv # personnal information of the patient in the vitalDB dataset
    ├─── create_dataset.py # download the needed data from vitaldb dataset
    ├─── separate_train_test.py # separate test and train set using the change of target as the critrion
    ├─── select_cases.py # filter the case from vitalDB to obtain the case_id of the patient used in the study
    ├─── all_reg.py # test different regressor for predicting BIS and MAP along the surgery
    ├─── delayed_pred.py # test the SVR regressor to predict BIS and MAP in the future for different time ahead
    ├─── svr_kernel.py # test different kernel for the SVR regressor
    ├─── standard_model.py # test the standard model for predicting BIS and MAP
    ├─── metrics_function.py # include function to compute prediction metrics and create usefull plot
    ├── LICENSE
    ├── requirements.txt
    ├── README.md
    └── .gitignore   
    
## Installation

Install all the required packages with the command:

```
pip install -r requirement.txt
```

## Usage
For standard model and regression evaluation:
 - First the database must be created from online vitalDB data. for this run the code *Create_dataset.py*
 - Then Standard models and Regressions technique can be evaluated by launching the corresponding scripts, hyperparameters are at the beginning of the script.
 
For Lee et al. model evaluation open *lee_code* folder:
  - First create the dedicated database (with only test cases) launching *Create_dataset_lee.py*
  - Launch the script *test_lee.py* to evaluate their method on our selected cases
  
## Authors

Bob Aubouin--Pairault, Mirko Fiacchini, Thao Dang

## License

MIT license
