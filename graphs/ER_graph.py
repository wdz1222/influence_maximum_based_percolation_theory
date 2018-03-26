# -*- coding: UTF-8 -*-

import numpy as np
from influence_index import *
import networkx as nx


class ERGraph:

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def create_er_graph(self):
        G = nx.fast_gnp_random_graph(self.n, self.p, seed=None, directed=False)
        for n, node in G.nodes_iter(data=True):
            node['trust'] = np.random.rand()
            node['distrust'] = np.random.rand()
        for e in G.edges_iter():
            G[e[0]][e[1]]['diffusivity'] = InfluenceIndex.calculate_diffusivity(G.node[e[0]], G.node[e[1]])
        graph_path = '../graph_data/er_graph_n%d_p%.1f.gpickle' % (self.n, self.p)
        nx.write_gpickle(G, graph_path)

if __name__ == "__main__":
    erg = ERGraph(50, 0.2)
    erg.create_er_graph()
