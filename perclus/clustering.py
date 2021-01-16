import numpy as np
from numba import njit


# Define UNION and FIND routines:
@njit
def find(x, labels):
    '''
    Finds the equivalence class of an element x in an array
    '''
    y = x
    # first follow the tree assigned to y to see its equivalence class (seed)
    while labels[y] != y:
        y = labels[y]
    # assign the label of y to all the tree (improves speed).
    while labels[x] != x:
        z = labels[x]  # store original pointer
        labels[x] = y  # relabel pointer
        x = z  # continue relabelling with original pointer
    return y


@njit
def union(x, y, labels):
    '''
    Make the seed of x equal to that of y and returns
    said class
    '''
    target = find(y, labels)
    labels[find(x, labels)] = target
    return target


@njit
def new_seed(labels):
    '''
    Creates a new equivalence class
    '''
    labels[0] += 1  # add to slot that counts No. of classes
    labels[labels[0]] = labels[0]  # condition that defines seed
    return labels[0]  # returns updated equivalence class label


@njit
def get_clusters(surface, open=False):
    '''
    Calculate clusters of the nodal surface (Using the
    Hoshen-Kopelman Algorithm)
    '''
    L = surface.shape[0]
    labels = np.zeros(L*L, dtype=np.int32)  # Assuming L*L equivalence classes

    for i in range(L):
        for j in range(L):
            if surface[i][j]:   # if active site
                up = (i > 0)*surface[i-1][j]  # upper boundary
                left = (j > 0)*surface[i][j-1]  # left boundary

                if up and left:
                    # add to an equivalence class
                    surface[i][j] = union(up, left, labels)
                if (up and not left) or (not up and left):
                    # put the nonzero label
                    surface[i][j] = max(up, left)
                if not up and not left:
                    # new cluster
                    surface[i][j] = new_seed(labels)

    # Periodic Boundary Conditions:
    if not open:
        for k in range(L):
            if surface[0][k] and surface[L-1][k]:
                union(surface[0][k], surface[L-1][k], labels)
            if surface[k][0] and surface[k][L-1]:
                union(surface[k][0], surface[k][L-1], labels)

    for i in range(L):
        for j in range(L):
            if surface[i][j]:
                # Relabel matrix so that only seeds are shown:
                surface[i][j] = find(surface[i][j], labels)

    return surface, labels


def get_masses(clusters, labels):
    masses = np.zeros(labels[0], np.int32)  # keep count of masses
    L = clusters.shape[0]
    for i in range(L):
        for j in range(L):
            if clusters[i][j]:
                x = find(clusters[i][j], labels)
                masses[x-1] += 1

    return masses[masses > 0]/L**2
