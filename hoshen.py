# HOSHEN KOPELMAN ALGORITHM FOR GENERATING CLUSTERS
###############################################################################

from numba import jit,prange
import numpy as np

#Define UNION and FIND functions:
################################### UNION-FIND ##########################################
@jit(nopython=True)
def find(x,label_array):  #find the equivalence class of x
    y=x
    #first follow the tree assigned to y to see its equivalence class (seed)
    while label_array[y] != y:
        y = label_array[y]
    #assign the label of y to all the tree.
    while label_array[x] != x:
        z = label_array[x]
        label_array[x] = y
        x = z
    return y

@jit(nopython=True)
def union(x,y,label_array):
#make the equivalence class of x equal to that of y and returns the equiv class
    label_array[find(x,label_array)] = find(y,label_array)
    return find(y,label_array)

@jit(nopython=True)
def make_set(label_array):
#create a new equivalence class
    label_array[0] +=1 #add to counter slot
    label_array[label_array[0]] = label_array[0] #assign equiv class
    return label_array[0]

@jit(nopython=True)
def switch(A,B):
    if A!=0 and B!=0:
        return 2
    if (A!=0 and B==0) or (A==0 and B!=0):
        return 1
    if A==0 and B==0:
        return 0

#Get clusters

################ Hoshen-Kopelman Algorithm ####################
@jit(nopython=True)
def get_clusters(surface):
    #calculate clusters of the nodal surface (Using the Hoshen-Kopelman Algorithm):
    L=len(surface[0])
    labels = np.array([0 for a in range(L**2) ]) #assume there's a max of L^2 equiv classes.

    for i in range(L):
        for j in range(L):
            if surface[i][j]>0:  #if occupied ...
                if i==0: up=0 #up=cluster_surf[L-1,j] #upper border
                else:    up=surface[i-1][j]
                #
                if j==0: left=0 #left=cluster_surf[i,L-1] #left border
                else:    left=surface[i][j-1]
                if switch(up,left)==0: surface[i][j] = make_set(labels)      #new cluster
                if switch(up,left)==1: surface[i][j] = max(up,left)    #whichever is nonzero is labelled
                if switch(up,left)==2: surface[i][j] = union(up,left,labels)  #add to an equivalence class

    #Periodic Boundary Conditions:
    for j in range(L):
        if surface[0][j]!= 0 and surface[L-1][j]!= 0:
            union(surface[0][j],surface[L-1][j],labels)
    for i in range(L):
        if surface[i][0]!= 0 and surface[i][L-1]!= 0:
            union(surface[i][0],surface[i][L-1],labels)
    #relabel matrix:
    for i in range(L):
        for j in range(L):
            if surface[i][j]>0:
                surface[i][j] = find(surface[i][j],labels)

    return surface
