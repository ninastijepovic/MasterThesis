import numpy as np


def labelsToClusters(labels, baselines):
    """
    Convert the labels that dbscan outputs to the format that is written to h5.
    :param labels: dbscan format, for instance: [0,0,1,0,1,2,2,1,-1] for 8 items in 3 clusters and 1 outlier.
    :param baselines: list of the baselines. Indices should correspond to the indices of labels.
    :return: dict, for instance {1: [('CS001', 'CS002'), ('CS001', 'CS003')]} for one cluster with 2 baselines.
    """
    nlabels = np.array(labels)
    nbaselines = np.array(baselines)
    clusters = {}
    for c in set(nlabels):
        mask = nlabels == c
        clusters[c] = [(a, b) for a, b in nbaselines[mask]]
    return clusters


def clusterToLabels(clusters, baselines):
    """
    Convert dict of clusters back to dbscan format.
    :param clusters: dict, for instance {1: [('CS001', 'CS002'), ('CS001', 'CS003')]} for one cluster with 2 baselines.
    :param baselines: list of the baselines. Indices should correspond to the indices of labels.
    :return: dbscan format, for instance: [0,0,1,0,1,2,2,1,-1] for 8 items in 3 clusters and 1 outlier.
    """
    labels = []
    for baseline in baselines:
        correct_cluster = findCluster(baseline, clusters)
        labels.append(correct_cluster)
    return labels


def findCluster(baseline, clusters):
    correct_cluster = None
    for cluster in clusters:
        if baseline in clusters[cluster]:
            correct_cluster = cluster
            continue
    return correct_cluster