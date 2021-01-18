
# import packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

#Let's create our lattice:
N = 48
G = nx.grid_2d_graph(N,N)
pos = dict( (n, n) for n in G.nodes() )

for i in range(N-1):
    for j in range(N-1):
        if np.random.random(1) > 0.5: G.remove_edge((i,j),(i+1,j))
        if np.random.random(1) > 0.5: G.remove_edge((i,j),(i,j+1))

# boundaries:
for k in range(N-1):
    G.remove_edge((N-1,k),(N-1,k+1))
    G.remove_edge((k,N-1),(k+1,N-1))

nx.draw_networkx(G, pos = pos, node_size = 0, with_labels = False)



plt.axis('off')
plt.show()
