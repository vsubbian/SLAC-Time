import time
import numpy as np
import faiss

__all__ = ['Kmeans', 'cluster_assign', 'arrange_clustering']

class ReassignedDataset:
    """A dataset where the new patients labels are given in argument.
    Args:
        patient_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to patients
        """

    def __init__(self, patient_indexes, pseudolabels, dataset):
        self.ptns = self.make_dataset(patient_indexes, pseudolabels, dataset)
        
    
    def make_dataset(self, patient_indexes, pseudolabels, dataset):
        patients = []
        for j, idx in enumerate(patient_indexes):
            patients.append(([dataset[i][idx] for i in range(4)], pseudolabels[j]))
        return patients
       

    def __len__(self):
        return len(patients)

def cluster_assign(patients_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        patients_lists (list of list): for each cluster, the list of patient indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset: a dataset with clusters as labels
    """
    assert patients_lists is not None
    pseudolabels = []
    patient_indexes = []
    for cluster, patients in enumerate(patients_lists):
        patient_indexes.extend(patients)
        pseudolabels.extend([cluster] * len(patients))

    return ReassignedDataset(patient_indexes, pseudolabels, dataset)

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on multiple GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
       
    clus = faiss.Clustering(d, nmb_clusters) 
    clus.seed = 42
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
   
    #CPU index
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def arrange_clustering(patients_lists):
    pseudolabels = []
    patient_indexes = []
    for cluster, patients in enumerate(patients_lists):
        patient_indexes.extend(patients)
        pseudolabels.extend([cluster] * len(patients))
    indexes = np.argsort(patient_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # cluster the data
        I, loss = run_kmeans(data, self.k, verbose)
        self.patients_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.patients_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
