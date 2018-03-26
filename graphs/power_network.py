# -*- coding: UTF-8 -*-

import numpy as np
from influence_index import *
import networkx as nx

class Power_network:

    def __init__(self, graph_path):
        self.G = nx.read_gml(graph_path)
        self.n = self.G.number_of_nodes()
        self.p = self.G.number_of_edges()

    def create_power_network(self):
        for n, node in self.G.nodes_iter(data=True):
            node['trust'] = np.random.rand()
            node['distrust'] = np.random.rand()
        for e in self.G.edges_iter():
            self.G[e[0]][e[1]]['diffusivity'] = InfluenceIndex.calculate_diffusivity(self.G.node[e[0]], self.G.node[e[1]])
        graph_path = '../graph_data/power_n%d_p%d.gpickle' % (self.n, self.p)
        nx.write_gpickle(self.G, graph_path)


if __name__ == "__main__":
    graph_path = '../graph_data/power.gml'
    pn = Power_network(graph_path)
    pn.create_power_network()
