# SLAC-Time

SLAC-Time is an innovative approach designed for clustering multivariate time-series data with missing values. This approach removes the necessity for data imputation and integration methods. The process uses a self-supervised Transformer to extract the pertinent representations from the input data. Clustering is then performed on these learned representations. Following this, the cluster assignments are utilized as labels to retrain the network. This process is iteratively repeated to continually improve the clusters.

The SLAC-Time folder comprises two key subfolders: `SLAC-Time_MultiMode` and `SLAC-Time_SingleMode`.

## SLAC-Time_MultiMode

This folder contains an implementation of SLAC-Time specifically designed to handle datasets incorporating both multivariate time-series data with missing values and non-temporal data. The code implements the SLAC-Time approach for clustering multivariate time-series data with missing values, as detailed in the research paper titled ["A self-supervised learning-based approach to clustering multivariate time-series data with missing values (SLAC-Time): An application to TBI phenotyping"](https://doi.org/10.1016/j.jbi.2023.104401).

## SLAC-Time_SingleMode

This subfolder includes code that implements another variant of SLAC-Time. This version is tailored to handle datasets composed exclusively of multivariate time-series data with missing values. The specifics of this implementation can be found in the paper ["Identifying TBI Physiological States by Clustering Multivariate Clinical Time-Series"](https://doi.org/10.48550/arXiv.2303.13024).

Please ensure to read the individual `README.md` files in each subfolder for specific instructions regarding the setup, data requirements, and execution of each implementation.
