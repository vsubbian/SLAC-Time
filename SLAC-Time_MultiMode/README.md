# SLAC-Time: A self-supervised learning-based approach to clustering multivariate time-series data with missing values
This code implements the SLAC-Time approach for clustering multivariate time series data with missing values, as described in the paper [A self-supervised learning-based approach to clustering multivariate time-series data with missing values (SLAC-Time): An application to TBI phenotyping](https://doi.org/10.1016/j.jbi.2023.104401). 

## Description
SLAC-Time is a novel approach developed for clustering multivariate time series data with missing values without resorting to data imputation and integration methods. It utilizes a self-supervised Transformer to extract representations from the input data, performs clustering on the learned representations, and then uses the cluster assignments as labels to retrain the network. This process is repeated iteratively to improve the clusters.

This particular implementation of SLAC-Time is designed to handle a dataset that includes both multivariate time series data with missing values and non-temporal data. Another version of SLAC-Time, presented in a separate work, is tailored to datasets comprising solely of multivariate time series data with missing values.

Specifically, this implementation follows these steps:

1. Preprocess data
    - Load raw data files
    - Clean the data
    - Normalize both non-temporal and time-series variables
    - Impute missing values of the non-temporal variables
    - Transform multivariate time series into observation triplets
2. Initialize deep neural network model
3. Pretrain model on forecasting task
4. Extract representations from pretrained model
5. Perform k-means clustering on representations
6. Assign cluster IDs as labels to the samples
7. Retrain neural network model using cluster labels
8. Repeat steps 4-7 for specified number of iterations
9. Save final cluster assignments

This document provides a guide to using the Python implementation of SLAC-Time, which is contained in four main Python files: `main.py`, `util.py`, `clustering.py` and `preprocess.py`. The code applies SLAC-Time to cluster patients with traumatic brain injury (TBI) using their multivariate clinical time series data. The goal is to identify TBI phenotypes based on temporal and non-temporal data.

## Code Files

**`main.py` script**: 
   * This file contains the main implementation of the SLAC-Time model. The script includes data loading, preprocessing, the building of the deep learning model, training of the model, and clustering of the samples.

**`preprocess.py` module**: 
  * This module handles data preprocessing, including loading, cleaning, and preparing the non-temporal and time-series data frames.

**`util.py` module**: 
   * This module contains utility functions that are used in the main script. These functions help with various tasks such as mapping list elements to their corresponding indices and checking if a string can be converted to a float.

**`clustering.py` module**: 
   * This module contains the implementation of K-means clustering used in the SLAC-Time approach.

## Key Classes and Functions

## File 1: `main.py`

**`CVE` class**: 
  * Custom layer for encoding continuous time and variable values without the need for discretization. The class inherits from tf.keras.layers.Layer and implements a call function that applies the layer operation to the input tensor.

**`Attention` class**: 
   * Implements attention mechanism that allows the model to focus on specific parts of the input data. Inherits from tf.keras.layers.Layer and implements a call function that computes the attention values.

**`Transformer` class**: 
   * Implements the transformer architecture, which is a type of attention mechanism. Inherits from tf.keras.layers.Layer and implements a call function that applies the transformer operation to the input tensor.

**`build_strats` function**: 
   * Builds the deep learning model using the custom classes. It utilizes instances of `CVE`, `Attention`, and `Transformer` classes to create a tf.keras model.

**`forecast_loss` function**: 
   * Custom loss function that calculates the loss for the forecasting task.

**`get_min_loss` function**: 
   * Auxiliary function that computes the minimum loss during training.


## File 2: `preprocess.py`

**`Connection` class**: 
   * Handles data loading and cleaning. The class has methods for loading multiple CSV files and cleaning them by handling missing values, transforming categorical variables, handling outliers, and normalizing features.

**`clean_clinic_data` function**: 
   * Handles the cleaning process for the non-temporal clinical data.

**`clean_vital_data` function**: 
   * Handles the cleaning process for the temporal vital data .

**`clean_lab_data` function**: 
   * Handles the cleaning process for the temporal lab data.

**`clean_gcs_data` function**: 
   * Handles the cleaning process for the temporal GCS data.

**`time_series` function**: 
   * Converts the cleaned data into time-series format.

**`vital_freq` function**: 
   * Calculates the frequency of vital signs in the data.


## File 3: `util.py`

**`inv_list` function**: 
   * Converts a list into a dictionary mapping the list elements to their indices.

**`isfloat` function**: 
   * checks if a given string can be converted to a float.

## File 4: `clustering.py`

1.**`ReassignedDataset` class**:
   * This class creates a new dataset where data points (patients in this case) are assigned new labels. It has two methods:
     * `__init__`: Initializes the class instance. Accepts `patient_indexes`, `pseudolabels`, and the `dataset`. The new dataset is created by using the `make_dataset` method.
     * `make_dataset`: Iterates through the `patient_indexes` and `pseudolabels` and selects corresponding patients from the `dataset`, thereby creating the new dataset.


2.**`cluster_assign` function**:
   * This function creates a new dataset based on the results of clustering, treating each cluster as a new label for its constituent data points (patients). It utilizes the `ReassignedDataset` class to achieve this.


3.**`run_kmeans` function**:
   * Executes k-means clustering on the data using the FAISS library. K-means is a clustering algorithm that segregates data into 'k' clusters, with each data point belonging to the cluster with the nearest mean. The function returns the cluster indices for each data point and the last clustering loss, indicating the performance of the clustering.


4.**`arrange_clustering` function**:
   * Accepts a list of patient lists where each list corresponds to a cluster. It reorders the `pseudolabels` and `patient_indexes` according to the original order of the patients in the dataset. This is essentially a utility function used to organize clustering results.


5.**`Kmeans` class**:
   * This class is a wrapper for executing k-means clustering on the given data. It comprises two main methods:
     * `__init__`: Initializes the class instance, simply storing the number of clusters (k) to be used in the k-means clustering.
     * `cluster`: Executes k-means clustering on the data using the `run_kmeans` function. It also records and prints the time taken for clustering (if `verbose=True`). The resulting clusters are saved in `self.patients_lists`.

## Dependencies

SLAC-Time implementation relies on the following Python libraries:

- numpy
- pandas
- tensorflow
- scikit-learn
- faiss
- tqdm
- time

## Data Source

The data utilized for this project is sourced from the TRACK-TBI dataset. This dataset comprises multivariate clinical time series data pertinent to patients diagnosed with Traumatic Brain Injury (TBI).

The raw dataset files should be stored under the `data/` directory. They are in four CSV files: `NSF U01 Clinical Data.csv`, `NSF U01 GCS and Pupillary Data.csv`, `NSF U01 Lab Data.csv`, and `NSF U01 Vitals Data.csv`. 

Upon preprocessing, the refined data is stored in `time_series_data.csv` for further use.

### Important Note

Due to NSF data sharing policies, the raw datasets are not included in this GitHub repository. To use this code, please obtain the dataset directly from the relevant authorities and place them in the appropriate directory as mentioned above.

## How to Execute the Script

To initiate the execution of the primary script, use the following command in your terminal:

```
python main.py
```

This will trigger the SLAC-Time workflow, which is designed to perform the clustering process.

The quantity of clusters `k` can be defined as a parameter within the `main.py` file.

After every iteration, the computed cluster assignments will be stored in the `cluster_assignments.npy` file.

Several hyperparameters including but not limited to batch size, learning rate, and the architecture of the neural network can be adjusted directly within the code as per your requirements.


## Reference

If you use this code, please cite the following paper:

Hamid Ghaderi, Brandon Foreman, Amin Nayebi, Sindhu Tipirneni, Chandan K. Reddy, Vignesh Subbian,
A self-supervised learning-based approach to clustering multivariate time-series data with missing values (SLAC-Time): An application to TBI phenotyping, Journal of Biomedical Informatics, Volume 143, 2023, 104401, ISSN 1532-0464. https://doi.org/10.1016/j.jbi.2023.104401.
