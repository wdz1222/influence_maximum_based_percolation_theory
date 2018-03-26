import networkx as nx
import numpy as np

class LTM:

    def __init__(self, graph_path, influencer_path, iter_num):
        self.G = nx.read_gpickle(graph_path)
        self.n = self.G.number_of_nodes()
        self.iter_num = iter_num
        self.diffusivity_ratio = np.zeros(self.iter_num, dtype=float)
        self.influence_index = self.init_influence_index(influencer_path)

    def init_influence_index(self, influencer_path):
        influencer = np.loadtxt(influencer_path, dtype=int)
        influence_index = np.zeros(self.n)
        influence_index[influencer] = 1
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
        print(self.diffusivity_ratio)

if __name__ == "__main__":
    graph_path = '../graph_data/er_graph_n50_p0.2.gpickle'
    influencer_path = '../influencer_data/er_graph_n50_p0_influencer_2.txt'
    ltm = LTM(graph_path, influencer_path, 50)
    ltm.LTM_algorithm(1)
    print(ltm.influence_index)