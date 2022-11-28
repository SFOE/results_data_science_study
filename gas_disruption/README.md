# Motivation

Due to the current geopolitical crisis, Switzerland is prone to higher risk of gas disruption. The goal of this project is to establish a gas disruption index computed on the last 10 years. This index should enable a monitoring of the risk of supply disruption over time and a comparison with neighbouring countries and their interdependencies. In addition, we would like to study this index under multiple scenario to better understand the impact of the gas supply.

# Installation

This package uses maily python. Make sure to have anaconda (or miniconda) installed on your computed. Then, you can install all the necessary dependencies by calling

> ```conda env create -f environment.yml```

and then activate the environment by calling

> ```conda activate gasindex```

# Usage
First run the ```preprocessing_data.ipynb``` notebook. It will preprocess the data that is used in the other 2 notebooks.

Then, we provide 2 notebooks for each type of analysis:

- ```simple_analysis.ipynb``` describes the gas disruption risk index using a simple approach,

- ```recurrent_approach.ipynb``` describes the gas disruption risk index using a recurrent definition of the risk in European countries. It takes into account the interactions between countries in Europe.

# Directory organization

```bash
├── code                     
│   ├── helpers.py                          # helping functions
│   ├── correction.py                       # bring correction from data
│   ├── preprocessing_data.ipynb            # preprocessing the data
│   ├── simple_analysis.ipynb               # Part 1: simple analysis
│   ├── recurrent_approach.ipynb            # Part 2: recursive formulation
│   ├── settings.py                         # settings 
├── data                                    # not shared
│   ├── preprocessed       
│   └── ...           
```

# Data

The data is obtained from Eurostat. Corrections from experts has been necessary.