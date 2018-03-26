import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# G = nx.fast_gnp_random_graph(50, 0.5, seed=None, directed=False)
# nx.draw(G, pos=nx.spring_layout(G), node_size=4, width=0.5)
# plt.show()
# G = nx.read_gml('graph_data/power.gml', )

# s = np.array([5, 5, 4, 3, 2])
# print np.where(s > 3)[0][-1]

G = nx.read_gpickle('graph_data/er_graph_n50_p0.2.gpickle')
print sum(G.degree())