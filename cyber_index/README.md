# Motivation

> The Switzerland has a quite diverse electrcity sector with approximately 600 large to small GRDs. Due to ongoing digitalization efforts, there is an increasing threat for cyber attacks. Hence, depending on the GRD criticality, there is a risk-based set of cyber security requirements for these companies. 
> The current method used by SFOE is only based on the population served and the tresholds are **fixed**. In this project we are interested in performing a **data-driven** classification of GRDs using the population served as well as additional inherent characteristics such as 
> - the critical infrastructure,
> - the power generation.

# Installation

This package uses maily python. Make sure to have anaconda (or miniconda) installed on your computed. Then, you can install all the necessary dependencies by calling

> ```conda env create -f environment.yml```

and then call

> ```conda activate cyberindex```

# Usage
First run the ```preprocessing_data.ipynb``` notebook. It will preprocess the data that is used in the other notebooks.

Then, we provide 4 notebooks:

- ```clustering.ipynb``` cluster the GRD into 4 categories. 
- ```comparison_cluster.ipynb``` tries to compare the clusters that were obtained.
- ```explaining_cluster.ipynb``` tries to explain the clusters using SHAPLEY values.
- ```map_vis.ipynb``` is a notebook to visualize with an interactive map the classification of the GRDs.

# Directory organization

```bash
├── code                     
│   ├── helpers.py                          # helping functions
│   ├── cluster_metric.py                   # metrics to score the clustering method
│   ├── preprocessing_data.ipynb            # preprocessing the data
│   ├── clustering.ipynb                    # Part 1: clustering
│   ├── comparison_cluster.ipynb            # Part 2: comparison of the clusters
│   ├── explaining_cluster.ipynb            # Part 3: explaining the classification using SHAP values
│   ├── map_vis.ipynb                       # Part 4: visualization of the clusters
│   ├── settings.py                         # settings 
├── data                                    # not shared    
│   └── ...           
├── clusters                                # not shared    
│   └── ...           
├── figure                                  # not shared    
│   └── ...           
```


# Data

All the data used in this project are open source. We used 
- a file containing diverse information/municipality (accessible at [here](https://www.bfs.admin.ch/bfs/fr/home/statistiques/catalogues-banques-donnees/tableaux.assetdetail.16484444.html))
- the list of municipality fusions since 2019 (obtained using [this](https://www.agvchapp.bfs.admin.ch/fr/communes/query) website)
- the taxable income/taxpayer/municipality from 2018 (accessible [here](https://www.atlas.bfs.admin.ch/maps/13/fr/16601_9164_9202_7267/25887.html#))
- the taxable income/capita/municipality from 2018 ([here](https://www.atlas.bfs.admin.ch/maps/13/fr/16602_9164_9202_7267/25889.html#))
- proportion of social subsidies in 2020 (accessible [here](https://www.atlas.bfs.admin.ch/core/projects/13/xshared/xlsx/25654_132.xlsx))
- the list of infrastructure:
    - We use overpass-turbo to retrieve the list of infrastructure en their GPS coordinates. We then use the municipality borders ([here](https://www.swisstopo.admin.ch/fr/geodata/landscape/boundaries3d.html)) and the software QGIS to count the number of infrastructure in each municipality.
- the local production is retrieved from opendata ([here](https://opendata.swiss/fr/dataset/elektrizitatsproduktionsanlagen)) and aggregated using QGIS
- the mapping between GRDs and municipalities ([here](https://www.elcom.admin.ch/dam/elcom/de/dokumente/2022/schweizerischegemeindenundzustaendigestromnetzbetreiberstand25.04.22.xlsx.download.xlsx/Schweizerische%20Gemeinden%20und%20zust%C3%A4ndige%20Stromnetzbetreiber.xlsx ))



Several files has been used to fill missing values:
- Filling missing values for economic features with 2019 data (use files from the swiss atlas [here](https://www.atlas.bfs.admin.ch/maps/13/fr/16419_9077_9075_138/25613.html))
- Filling missing values with 2019/2020 population ([here](https://www.bfs.admin.ch/bfs/fr/home/statistiques/catalogues-banques-donnees/tableaux.assetdetail.11587766.html))
