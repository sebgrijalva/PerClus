# HOSHEN KOPELMAN ALGORITHM FOR GENERATING CLUSTERS
###############################################################################

from numba import njit
import numpy as np

# Define UNION and FIND functions:
################################### UNION-FIND ##########################################
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
        z = labels[x] # store original pointer
        labels[x] = y # relabel pointer
        x = z # continue relabelling with original pointer
    return y

@njit
def union(x, y, labels):
    '''
    Make the seed of x equal to that of y and returns
    said class
    '''
    labels[find(x, labels)] = find(y, labels)
    return find(y,labels)

@njit
def new_seed(labels):
    '''
    Creates a new equivalence class
    '''
    labels[0] += 1 # add to slot that counts No. of classes
    labels[labels[0]] = labels[0] # condition that defines seed
    return labels[0] # returns updated equivalence class label

#Get clusters

################ Hoshen-Kopelman Algorithm ####################
@njit
def get_clusters(surface, periodic = True):
    '''
    Calculate clusters of the nodal surface (Using the Hoshen-Kopelman Algorithm)
    '''
    L = surface.shape[0]
    labels = np.zeros(L*L, dtype=np.int8) # Assuming L*L equivalence classes

    for i in range(L):
        for j in range(L):
            up, left = 0,0 # assume they are both empty at first
            if surface[i][j]:  #if occupied
                if i: up = surface[i-1][j] # upper boundary
                if j: left = surface[i][j-1] # left boundary
                #
                if up and left: surface[i][j] = new_seed(labels)      #new cluster
                if (up and not left) or (not up and left): surface[i][j] = max(up, left)  #put the nonzero label
                if not up and not left: surface[i][j] = union(up, left, labels)  #add to an equivalence class

    # Periodic Boundary Conditions:
    if periodic:
        for k in range(L):
            if surface[0][k] and surface[L-1][k]:
                union(surface[0][k], surface[L-1][k], labels)
            if surface[k][0] and surface[k][L-1]:
                union(surface[k][0], surface[k][L-1], labels)

    # Relabel matrix so that only seeds are shown:
    for i in range(L):
        for j in range(L):
            if surface[i][j]:
                surface[i][j] = find(surface[i][j], labels)

    return surface
