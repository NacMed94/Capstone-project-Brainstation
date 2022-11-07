# Capstone-project-Brainstation

Predicting hospital occupancy by department based on current patients using MIMIC IV dataset

Prescription table contains categorisation into ATC system (level 1 & 2) of prescriptions from tables medrecon (from emergency department dataset) and prescriptions (hospital dataset).

Merging tables is the creation of the final table, including data cleaning and some feature extraction.

EDA contains the EDA and feature engineering.

Model optimisation notebook contains pipelines and model optimizations of logistic regression and random forests.

The folder ndc_mapping contains the NDC list (unique_NDC.csv) mapped into ndc_map 2022_10_12 17_12 (atc4).csv by the R script ndc_map. Script accessible here:

https://github.com/fabkury/ndc_map

Data used in study not available, please contact me for more info