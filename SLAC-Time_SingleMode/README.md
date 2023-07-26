# Identifying TBI Physiological States by Clustering Multivariate Clinical Time-Series
This code implements another version of SLAC-Time that addresses a dataset comprising solely of multivariate time series data with missing values, as described in the paper [Identifying TBI Physiological States by Clustering Multivariate Clinical Time-Series](https://doi.org/10.48550/arXiv.2303.13024).

## Description
SLAC-Time is a novel approach developed for clustering multivariate time series data with missing values without resorting to data imputation and integration methods. It utilizes a self-supervised Transformer to extract representations from the input data, performs clustering on the learned representations, and then uses the cluster assignments as labels to retrain the network. This process is repeated iteratively to improve the clusters.

Specifically, this implementation follows these steps:

1. Preprocess data
    - Load raw data files
    - Clean the data
    - Normalize time-series variables
    - Transform multivariate time series into observation triplets
2. Initialize deep neural network model
3. Pretrain model on forecasting task
4. Extract representations from pretrained model
5. Perform k-means clustering on representations
6. Assign cluster IDs as labels to the samples
7. Retrain neural network model using cluster labels
8. Repeat steps 4-7 for specified number of iterations
9. Save final cluster assignments

This document provides a guide to using the Python implementation of this version of SLAC-Time, which is contained in four main Python files: `main.py`, `preprocess.py`, `util.py`, and `clustering.py`. The code applies SLAC-Time to cluster multivariate clinical time series data. The goal of implementation is to identify physiological states of the patients with Traumatic Brain Injury (TBI) based on the clinical multivariate time-series data.

**Note**: Due to NSF data privacy regulations, we are unable to provide the actual code for `preprocess.py`. This module contains sensitive information such as the actual identifiers of the patients.


## Code Files

**`main.py` script**: 
   * This file contains the main implementation of the SLAC-Time model. The script includes data loading, preprocessing, the building of the deep learning model, training of the model, and clustering of the samples.

**`preprocess.py` module**: 
  * This module handles data preprocessing, including loading and cleaning data.

**`util.py` module**: 
   * This module contains utility functions that are used in the main script. These functions help with various tasks such as mapping list elements to their corresponding indices and checking if a string can be converted to a float.
   
**`clustering.py` module**: 
* This module contains the implementation of K-means clustering used in the SLAC-Time approach.

## Key Classes and Functions

### `main.py`

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


### `util.py`



**`extr_org_sample` function**: 
  * extract a multivariate time series for a specific patient from a complex 3D DataFrame.

**`TimeSeriesScaler` class**: 
  * Handles scaling of the time series data.

**`inv_list` function**: 
  * Converts a list into a dictionary mapping the list elements to their indices.

**`isfloat` function**: 
  * Checks if a given string can be converted to a float.


## `clustering.py`

1.**`ReassignedDataset` class**:
   * This class creates a new dataset where samples are assigned new labels. It has two methods:
     * `__init__`: Initializes the class instance. Accepts `sample_indexes`, `pseudolabels`, and the `dataset`. The new dataset is created by using the `make_dataset` method.
     * `make_dataset`: Iterates through the `sample_indexes` and `pseudolabels` and selects corresponding samples from the `dataset`, thereby creating the new dataset.


2.**`cluster_assign` function**:
   * This function creates a new dataset based on the results of clustering, treating each cluster as a new label for its constituent samples. It utilizes the `ReassignedDataset` class to achieve this.


3.**`run_kmeans` function**:
   * Executes k-means clustering on the data using the FAISS library. K-means is a clustering algorithm that segregates data into 'k' clusters, with each sample belonging to the cluster with the nearest mean. The function returns the cluster indices for each sample and the last clustering loss, indicating the performance of the clustering.


4.**`arrange_clustering` function**:
   * Accepts a list of sample lists where each list corresponds to a cluster. It reorders the `pseudolabels` and `sample_indexes` according to the original order of the samples in the dataset. This is essentially a utility function used to organize clustering results.


5.**`Kmeans` class**:
   * This class is a wrapper for executing k-means clustering on the given data. It comprises two main methods:
     * `__init__`: Initializes the class instance, simply storing the number of clusters (k) to be used in the k-means clustering.
     * `cluster`: Executes k-means clustering on the data using the `run_kmeans` function. It also records and prints the time taken for clustering (if `verbose=True`). The resulting clusters are saved in `self.samples_lists`.


## Dependencies

This version of SLAC-Time implementation relies on the following Python libraries:

- numpy
- pandas
- tensorflow
- scikit-learn
- faiss
- tqdm
- time
- os
- scipy
- gc



## Data Source

The data utilized for this project is sourced from the TRACK-TBI dataset. Specifically, our investigation focused on 16 patients with Traumatic Brain Injury (TBI) for whom high-resolution recordings of physiological data were available.

Upon preprocessing, the required data for the clustering purpose is stored in a three-dimensional numpy array named "value". In this array, the first dimension is related to the samples, the second dimension is associated with time, and the third dimension is related to the variables. Each element in the numpy array "value" corresponds to the value of a variable at a specific time for a specific time-series sample.


### Important Note

For full execution of this code, you will need to develop your own `preprocess.py` module suitable for your data. This module should perform the necessary data loading, cleaning, and preparation tasks to make your data in the form of the "value" numpy array for the SLAC-Time model.


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

Hamid Ghaderi, Brandon Foreman, Amin Nayebi, Sindhu Tipirneni, Chandan K. Reddy, Vignesh Subbian, “Identifying TBI Physiological States by Clustering of Multivariate Clinical Time-Series,” Mar. 2023, Accessed: Jun. 20, 2023. [Online]. Available: https://arxiv.org/abs/2303.13024v2.

---



```python

```
