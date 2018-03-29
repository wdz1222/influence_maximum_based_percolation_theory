# -*- coding: UTF-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Graph:

    def __init__(self, graph_path):
        graph_type = graph_path.split('/')[-1].split('.')[-1]
        print(graph_type)
        if graph_type == 'gpickle':
            self.G = nx.read_gpickle(graph_path)
        elif graph_type == 'gml':
            self.G = nx.read_gml(graph_path)

    def paintGraph(self, influencer_path, threshold_path, threshold):
        influencer = np.loadtxt(influencer_path, dtype=int)
        ts = np.loadtxt(threshold_path)
        label = ['r']*self.G.number_of_nodes()
        label = np.array(label)
        print(influencer[0:np.where(ts>threshold)[0][-1]])
        label[influencer[0:np.where(ts > threshold)[0][-1]]] = 'y'
        nx.draw(self.G, pos=nx.spring_layout(self.G), node_size=10, width=0.3, node_color=label, with_labels= False)
        plt.show()

    def paint_eig(self, threshold_path1, threshold_path2, threshold_path3):
        ts1 = np.loadtxt(threshold_path1)
        ts2 = np.loadtxt(threshold_path2)
        ts3 = np.loadtxt(threshold_path3)
        n = self.G.number_of_nodes()
        x = (np.array(range(n))+1)/float(n)
        plt.figure(1)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(x, ts1, linestyle='-')
        plt.plot(x, ts2, linestyle='--')
        plt.plot(x, ts3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 8)
        label = ["union_order=1", "union_order=2", "union_order=3", u'阈值比例', u'特征值阈值']
        plt.legend(label, loc=1, ncol=1)
        plt.show()

if __name__ == "__main__":
    # graph_path = '../graph_data/er_graph_n50_p0.2.gpickle'
    # graph = Graph(graph_path)
    # influencer_path = '../influencer_data/er_graph_n50_p0_influencer_3.txt'
    # threshold_path = '../influencer_data/er_graph_n50_p0_eig_3.txt'
    # threshold = 1.26096
    # Graph.paintGraph(graph_path, influencer_path, threshold_path, threshold)

    # threshold_path1 = '../influencer_data/er_graph_n50_p0_eig_1.txt'
    # threshold_path2 = '../influencer_data/er_graph_n50_p0_eig_2.txt'
    # threshold_path3 = '../influencer_data/er_graph_n50_p0_eig_3.txt'
    # graph.paint_eig(threshold_path1, threshold_path2, threshold_path3)

    graph_path = '../graph_data/netscience_n1589_p2742.gpickle'
    graph = Graph(graph_path)
    influencer_path = '../influencer_data/netscience_n1589_p2742_influencer_3.txt'
    threshold_path = '../influencer_data/netscience_n1589_p2742_eig_3.txt'
    threshold = 0.3617
    graph.paintGraph(influencer_path, threshold_path, threshold)

    # threshold_path1 = '../influencer_data/netscience_n1589_p2742_eig_1.txt'
    # threshold_path2 = '../influencer_data/netscience_n1589_p2742_eig_2.txt'
    # threshold_path3 = '../influencer_data/netscience_n1589_p2742_eig_3.txt'
    # threshold = [0.0471, 0.217496, 0.3617]
    # graph.paint_eig1(threshold_path1, threshold_path2, threshold_path3, threshold)