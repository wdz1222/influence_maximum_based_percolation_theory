# -*- coding: UTF-8 -*-

import networkx as nx
import numpy as np


class Other:

    def __init__(self, graph_path):
        self.G = nx.read_gpickle(graph_path)
        self.G_name = graph_path.split('/')[-1].split('.')[0]

    def closeness_vitality(self):
        cv = nx.closeness_vitality(self.G, weight='diffusivity')
        result = Other.sort_result(cv)
        result_path = '../other_data/%s_closeness.txt' % self.G_name
        np.savetxt(result_path, result, fmt='%d')

    def degree_centrality(self):
        dc = nx.degree_centrality(self.G)
        print(dc)
        result = Other.sort_result(dc)
        result_path = '../other_data/%s_degree_centrality.txt' % self.G_name
        np.savetxt(result_path, result, fmt='%d')

    def eigenvector_centrality(self):
        for e in self.G.edges_iter():
            self.G[e[0]][e[1]]['diffusivity_tmp'] = 1 - self.G[e[0]][e[1]]['diffusivity']
        ec = nx.eigenvector_centrality(self.G, weight='diffusivity_tmp')
        result = Other.sort_result(ec)
        result_path = '../other_data/%s_eigenvector_centrality.txt' % self.G_name
        np.savetxt(result_path, result, fmt='%d')

    @staticmethod
    def sort_result(dic):
        dictr = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        result = np.array(map(lambda x: x[0], dictr))
        return result

if __name__ == "__main__":
    other = Other('../graph_data/er_graph_n50_p0.2.gpickle')
    # other.closeness_vitality()
    # other.degree_centrality()
    other.eigenvector_centrality()