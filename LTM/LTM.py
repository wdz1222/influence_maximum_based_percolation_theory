import networkx as nx
import numpy as np

class LTM:

    def __init__(self, graph_path, influencer_path, iter_num, seeds_num):
        self.G = nx.read_gpickle(graph_path)
        self.n = self.G.number_of_nodes()
        self.iter_num = iter_num
        self.seeds_num = seeds_num
        self.diffusivity_ratio = np.zeros(self.iter_num, dtype=np.float)
        self.influence_index = self.init_influence_index(influencer_path)

    def init_influence_index(self, influencer_path):
        influencer = np.loadtxt(influencer_path, dtype=np.int)
        if len(influencer.shape) != 1:
            influencer = influencer[:, 0]
        print(influencer)
        influence_index = np.zeros(self.n)
        influence_index[influencer[0:self.seeds_num]] = 1
        return influence_index

    def LTM_algorithm(self, active_threshold):
        influence_index_temp = list()
        for j in range(self.iter_num):
            for i in range(self.n):
                weight = 0
                if self.influence_index[i] == 0:
                    friends = self.G[i]
                    for key in friends.keys():
                        if self.influence_index[key] == 1:
                            weight = weight + (1-friends[key]['diffusivity'])
                    if weight > active_threshold:
                        influence_index_temp.append(i)
            self.influence_index[influence_index_temp] = 1
            self.diffusivity_ratio[j] = sum(self.influence_index)/self.n
            # print(self.influence_index)
        print(self.diffusivity_ratio)


if __name__ == "__main__":
    # graph_path = '../graph_data/er_graph_n50_p0.2.gpickle'
    # influencer_path = '../influencer_data/er_graph_n50_p0_influencer_3.txt'
    # influencer_path = '../other_data/er_graph_n50_p0_closeness.txt'
    # influencer_path = '../other_data/er_graph_n50_p0_degree_centrality.txt'
    # influencer_path = '../other_data/er_graph_n50_p0_eigenvector_centrality.txt'

    graph_path = '../graph_data/netscience_n1589_p2742.gpickle'
    # influencer_path = '../influencer_data/netscience_n1589_p2742_influencer_3.txt'
    # influencer_path = '../other_data/netscience_n1589_p2742_closeness.txt'
    # influencer_path = '../other_data/netscience_n1589_p2742_degree_centrality.txt'
    influencer_path = '../other_data/netscience_n1589_p2742_eigenvector_centrality.txt'
    ltm = LTM(graph_path, influencer_path, 50, 500)
    ltm.LTM_algorithm(0.2)
    print(ltm.influence_index)