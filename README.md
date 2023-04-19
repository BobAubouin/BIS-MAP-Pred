# BIS-MAP-Pred
GitHub associated with the research paper "Data-based Pharmacodynamic Modeling for BIS and Mean Arterial Pressure Prediction during General Anesthesia"

## Installation

Install all the required packages with the command:

```python
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
