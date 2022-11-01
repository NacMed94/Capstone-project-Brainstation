# Capstone-project-Brainstation

Predicting hospital occupancy by department based on current patients using MIMIC IV dataset

Prescription table contains categorisation into ATC system (level 1) of prescriptions from tables medrecon (from emergency department dataset) and prescriptions (hospital dataset). Possible second stage to increase granularity (go down levels) of prescriptions if necessary

Merging tables is the creation of the final table. Only one stage. Includes some feature extraction (extracted pickle of past transfers)

EDA (will) contain the EDA and feature selection. This is done in stages. First stage completed with a first set of features ready to try on models. A second and possibly third stage will be added to add and characterise new features (previous transfer location and granularity increase of prescriptions columns).

Model optimisation notebook contains pipelines and model descriptions of different models. New features will be added as needed.
