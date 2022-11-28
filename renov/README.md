# Motivation

The SFOE has a database containing more than 70000 entries concerning supported energy efficiency measures between 2017 and 2021. The goal of this project is to understand were this subsidies are used and by who. In addition, we want to understand which measures are the most effective in CO2 reduction and in kWh production.
This project should help reporting the results and provide insights how the population is sensitive to the supported measures.

# Installation
This package uses maily python. Make sure to have anaconda (or miniconda) installed on your computed. Then, you can install all the necessary dependencies by calling
> ```conda env create -f environment.yml``` 

and then call

> ```conda activate renov```

# Usage

First run the ```preprocessing_data.ipynb``` notebook. It will preprocess the data that is used in the other 3 notebooks. 

Then, we provide 3 notebooks for each type of analysis:
- ```region_subsidy.ipynb``` describes how the subsidies are distributed among regions in CH.
- ```effective_measure_analysis.ipynb``` describes which supported measures are used and the cost of them.
- ```population_analysis.ipynb``` attempts to understand which population benefits from the subsidies. At this stage only an analysis at the municipality level is performed as no information about the owners could be found.


# Directory organization

```bash
├── code                     
│   ├── helpers.py                          # helping functions
│   ├── plot_maps.py                        # plotting maps
│   ├── preprocessing_data.ipynb            # preprocessing the data
│   ├── region_subsidy.ipynb                # Part 1: subsidy per region
│   ├── effective_measure_analysis.ipynb    # Part 2: subsidy per measure
│   ├── population_analysis.ipynb           # Part 3: subsidy per pop.
├── data                                    # not shared
│   ├── preprocessed       
│   └── ...           
```


# Data 

- The main database is private and is owned by the SFOE.
- In addition we use data from the Swiss federal office of statistics (FSO) ([here](https://www.atlas.bfs.admin.ch/maps/13/fr/16894_72_71_70/26207.html)) and the Federal register of buildings and dwellings (RegBl, [here](https://www.bfs.admin.ch/bfs/fr/home/registres/registre-batiments-logements.html)).


## Authors and acknowledgment
This project is a collaboration with the Swiss federal offfice of energy (SFOE) and the Swiss data science center (SDSC). 