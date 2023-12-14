import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

def test_mean_hamming_distance(): 
    permutations = np.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 4, 3],
                             [0, 2, 1, 3, 4],
                             [0, 2, 1, 4, 3],
                             [0, 1, 3, 2, 4]])
    distance_matrix = cdist(permutations, permutations, metric="hamming")

    print("Distance Matrix: \n ", distance_matrix)
    upper_diag = distance_matrix[np.triu_indices(distance_matrix.shape[0])]

    print("Upper diag: \n", upper_diag)
    print("Upper diag mean: ", upper_diag.mean())

def test_probability_utilization():

    permutations = np.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 4, 3],
                             [0, 2, 1, 3, 4],
                             [0, 2, 1, 4, 3],
                             [0, 1, 3, 2, 4]])
    distance_matrix = cdist(permutations, permutations, metric="hamming")

    print("Distance Matrix: \n ", distance_matrix)
    upper_diag = distance_matrix[np.triu_indices(distance_matrix.shape[0])]

    print("Upper diag: \n", upper_diag)
    print("Upper diag mean: ", upper_diag.mean())
    print("Upper diag std: ", upper_diag.std())


    operation_indices = np.array([0, 1, 2, 3])
    print("Operation indices: ", operation_indices)
    
    scaled_indices = (operation_indices - operation_indices.min()) / (operation_indices.max() - operation_indices.min())
    print("Scaled indices: ", scaled_indices)

    operation_probs = norm.pdf(scaled_indices, loc=upper_diag.mean(), scale=upper_diag.std())
    print("operation_probs: ", operation_probs.round(4))

    prob = np.random.normal(loc=upper_diag.mean(), scale=upper_diag.std())
    print("Sample: ", prob)


if __name__ == "__main__":
    #test_mean_hamming_distance()
    test_probability_utilization()