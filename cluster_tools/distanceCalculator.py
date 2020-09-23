from scipy.spatial.distance import pdist, squareform


def get_swapped(baseline):
    return baseline[1], baseline[0]


def get_alternative_indices(i, index_by_baseline, baselines):
    original = baselines[i]
    swapped = get_swapped(original)
    alternatives = [index_by_baseline[s] for s in [swapped] if s in index_by_baseline]
    return alternatives


def get_reduced_distances(distances, baselines):
    indices_by_baseline = {baseline: i for i, baseline in enumerate(baselines)}

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            indices_i = [i] + get_alternative_indices(i, indices_by_baseline, baselines)
            indices_j = [j] + get_alternative_indices(j, indices_by_baseline, baselines)
            pairs = []
            for index_i in indices_i:
                for index_j in indices_j:
                    pairs.append((index_i, index_j))
            distances[i, j] = min([distances[pair[0], pair[1]] for pair in pairs])
    return distances


def get_distance_matrix(data, baselines):
    if data is None:
        raise ValueError('Data cannot be None.')

    if baselines is None:
        raise ValueError('number_of_stations cannot be None.')

    if len(data.shape) != 2:
        raise ValueError('Data should contain one or multiple feature columns for each row.')

    if data.shape[0] != len(baselines):
        raise ValueError('Number of data rows should be equal to number of stations squared.')

    distances = squareform(pdist(data))
    return get_reduced_distances(distances, baselines)
