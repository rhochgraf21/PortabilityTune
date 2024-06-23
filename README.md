# Portability Tuning

Base code for the Portability Tuning project.

## Repo Structure

In each main folder, there is a README describing the items in that folder.

The folder contents are as follows:

- CLBlast: a copy of CLBlast's source that is modified to allow runtime selection of pre-compiled GEMM kernels 
- datasets: JSON data files containing performance results collected from the remote machines (our data)
- graph_scripts: scripts for generating the (majority of the) graphs used in the paper manuscript
- helper_scripts: helper scripts used for data collection, preprocessing, and validation
- other: misc reference items
- ptuner: core source code for running portability tunings
- tuning_kernels: moderately modified source code from jwlawson; dependency for performing alternative ways of portability tuning (k-means/decision tree)