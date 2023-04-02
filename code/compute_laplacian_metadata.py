"""Compute the metadata used for the laplacian loss term in losses.py"""
import numpy as np
import pandas as pd

NUM_FACTORS = 4

ch_names = ['fp1', 'f3', 'f7', 'c3', 't7', 'p3', 'p7', 'o1', 
            'fp2', 'f4', 'f8', 'c4', 't8', 'p4', 'p8', 'o2',
            'fz', 'cz', 'pz']

df = pd.read_csv("../data/standard_1020_3D.tsv", sep = "\t")

labels = df.values[:,0].flatten()
mask = [(label.lower() in ch_names) for label in labels]

masked_labels = labels[mask]

coordinates = df.values[:,1:]

masked_coordinates = coordinates[mask]

coord_matrix = masked_coordinates @ masked_coordinates.T

coord_matrix = coord_matrix.astype(float)

coord_matrix = np.arccos(coord_matrix) 

max_indices = np.argsort(coord_matrix, axis=1)[:, 1: 1 + NUM_FACTORS]

distance_matrix = np.zeros(coord_matrix.shape)

for i in range(len(max_indices)):
    distance_matrix[i,max_indices[i]] = 1/coord_matrix[i,max_indices[i]]

dist_sum = np.sum(distance_matrix, axis = 1)

distance_matrix = distance_matrix / dist_sum

permute_labels = []

for ch in ch_names:
    for i in range(len(masked_labels)):
        if ch == masked_labels[i].lower():
            permute_labels.append(i)
            break

distance_matrix = distance_matrix[permute_labels]
distance_matrix = distance_matrix[:, permute_labels]

distance_matrix[distance_matrix > 0] = 1 / NUM_FACTORS

print(distance_matrix)

np.save("../data/distance_matrix.npy", distance_matrix)



