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
    target = find(y, labels)
    labels[find(x, labels)] = target
    return target

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
def get_clusters(surface, open=False, get_masses=False):
    '''
    Calculate clusters of the nodal surface (Using the Hoshen-Kopelman Algorithm)
    '''
    L = surface.shape[0]
    labels = np.zeros(L*L, dtype=np.int32) # Assuming L*L equivalence classes

    for i in range(L):
        for j in range(L):
            if surface[i][j]:  #if active site
                up = (i>0)*surface[i-1][j] # upper boundary
                left = (j>0)*surface[i][j-1] # left boundary
                #
                if up and left: surface[i][j] = union(up, left, labels)  #add to an equivalence class
                if (up and not left) or (not up and left): surface[i][j] = max(up, left)  #put the nonzero label
                if not up and not left: surface[i][j] = new_seed(labels)      #new cluster

    # Periodic Boundary Conditions:
    if not open:
        for k in range(L):
            if surface[0][k] and surface[L-1][k]:
                union(surface[0][k], surface[L-1][k], labels)
            if surface[k][0] and surface[k][L-1]:
                union(surface[k][0], surface[k][L-1], labels)

    # Relabel matrix so that only seeds are shown:
    for i in range(L):
        for j in range(L):
            if surface[i][j]:
                surface[i][j] = find(surface[i][j],labels)
    return surface


@njit
def get_masses(surface, open=False):
    '''
    Calculate clusters of the nodal surface (Using the Hoshen-Kopelman Algorithm)
    '''
    L = surface.shape[0]
    labels = np.zeros(L*L, dtype=np.int32) # Assuming L*L equivalence classes

    for i in range(L):
        for j in range(L):
            if surface[i][j]:  #if active site
                up = (i>0)*surface[i-1][j] # upper boundary
                left = (j>0)*surface[i][j-1] # left boundary
                #
                if up and left: surface[i][j] = union(up, left, labels)  #add to an equivalence class
                if (up and not left) or (not up and left): surface[i][j] = max(up, left)  #put the nonzero label
                if not up and not left: surface[i][j] = new_seed(labels)      #new cluster

    # Periodic Boundary Conditions:
    if not open:
        for k in range(L):
            if surface[0][k] and surface[L-1][k]:
                union(surface[0][k], surface[L-1][k], labels)
            if surface[k][0] and surface[k][L-1]:
                union(surface[k][0], surface[k][L-1], labels)

    # Count masses of clusters:
    masses = np.zeros(labels[0],np.int32) #keep count of masses
    for i in range(L):
        for j in range(L):
            if surface[i][j]:
                x = find(surface[i][j],labels)
                masses[x-1] += 1
    return masses[masses>0]/L**2

'''
## Example:
import matplotlib.pyplot as plt
L = 10
surf = np.random.randint(0,3,(L,L))
plt.matshow(surf)
clusters,masses = get_clusters(surf,open=True)
print(masses)
plt.matshow(clusters)
plt.show()
'''
