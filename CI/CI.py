# -*- coding: UTF-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class CI:

    def __init__(self, graph_path, unite_order):
        self.G = nx.read_gpickle(graph_path)
        self.unite_order = unite_order
        self.G_name = graph_path.split('/')[-1].split('.')[0]
        self.n = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()
        self.w0 = self.caculate_w0()
        self.influence_index = np.zeros(self.n) + 1
        self.dig_edges, self.non_backtracking_matrix, self.non_backtracking_init_matrix = self.create_non_backtracking_matrix()
        self.max_eigval = self.estimate_matrix_max_eigval()
        self.max_init_eigval = self.estimate_init_matrix_max_eigval()
        self.influence_threshold = self.caculate_influence_number()

    def caculate_influence_number(self):
        return self.max_init_eigval/self.max_eigval

    def create_non_backtracking_matrix(self):
        DiG = self.G.copy().to_directed()
        dig_edges_number = DiG.number_of_edges()
        dig_edges = DiG.edges()
        non_backtracking_matrix = np.zeros([dig_edges_number, dig_edges_number])
        non_backtracking_init_matrix = np.zeros([dig_edges_number, dig_edges_number])
        for i in range(dig_edges_number):
            e_row = dig_edges[i]
            for j in range(dig_edges_number):
                e_col = dig_edges[j]
                if e_row[0] != e_col[1] and e_row[1] == e_col[0]:
                    non_backtracking_matrix[i, j] = self.calculate_removed_weight(e_row)
                    non_backtracking_init_matrix[i, j] = 1
        print('successful construct non-backtracking maxtrix')
        return dig_edges, non_backtracking_matrix, non_backtracking_init_matrix

    def ci_algorithm(self):
        assert self.unite_order > 0 and isinstance(self.unite_order, int)
        ci_index = np.zeros(self.n)
        eig = list()
        index = list()
        while sum(self.influence_index) != 0:
            for i in range(self.n):
                if self.influence_index[i] == 1:
                    ci_index[i] = self.caculate_ci_index(i)
            maxci = np.argmax(ci_index)
            print('ci_index=', np.max(ci_index))
            self.influence_index[maxci] = 0
            ci_index[maxci] = -1
            self.remove_influence_node(maxci)
            max_eigval_current = self.estimate_matrix_max_eigval()
            eig.append(max_eigval_current)
            index.append(maxci)
            print('max_eigval_current=', max_eigval_current)
        self.save_influencer(self.unite_order, eig, index)

    def remove_influence_node(self, i):
        self.G.remove_node(i)
        for j in range(len(self.dig_edges)):
            if self.dig_edges[j][1] == i:
                self.non_backtracking_matrix[j, :] = np.zeros(len(self.dig_edges))

    def caculate_ci_index(self, i):
        ci_index = 0
        unite_order_friends = self.search_unite_order_friends(i)
        for j in unite_order_friends:
            ci_index += self.caculate_i_weight(j)
        ci_index = ci_index * self.caculate_i_weight(i)
        return ci_index

    def calculate_init_ci_index(self, i):
        ci_index = 0
        unite_order_friends = self.search_unite_order_friends(i)
        for j in unite_order_friends:
            ci_index += self.G.degree(j)
        ci_index = ci_index * self.G.degree(i)
        return ci_index

    def search_unite_order_friends(self, i):
        if self.unite_order == 1:
            unite_order_friends = self.G.neighbors(i)
        else:
            unite_order_temp = 2
            unite_order_friends_temp = list()
            unite_order_friends_temp.extend(self.G.neighbors(i))
            unite_order_friends_temp.append(-1)
            while unite_order_temp <= self.unite_order:
                friendi = unite_order_friends_temp.pop(0)
                while friendi != -1:
                    unite_order_friends_temp.extend(self.G.neighbors(friendi))
                    friendi = unite_order_friends_temp.pop(0)
                unite_order_friends_temp.append(-1)
                unite_order_temp += 1
            unite_order_friends_temp.pop()
            unite_order_friends = set(unite_order_friends_temp)
            unite_order_friends.discard(i)
        return unite_order_friends

    def caculate_i_weight(self, i):
        dic_diffusivity = self.G[i]
        weight = 0
        for key in dic_diffusivity.keys():
            weight += dic_diffusivity[key]['diffusivity']
        return weight

    def calculate_removed_weight(self, e_row):
        dic_diffusivity = self.G[e_row[0]]
        weight = 0.0
        friend_number = len(self.G.neighbors(e_row[0])) - 1
        if friend_number != 0:
            for key in dic_diffusivity.keys():
                if key == e_row[1]:
                    continue
                else:
                    weight += dic_diffusivity[key]['diffusivity']
            weight = weight/friend_number
        else:
            weight = 1
        return weight

    @staticmethod
    def caculate_matrix_max_eigval(matrix):
        eigenvalues, eigenvalues_vectors = np.linalg.eig(matrix)
        maxeigval = -1
        for eigenvalue in eigenvalues:
            if np.imag(eigenvalue) == 0:
                if eigenvalue > maxeigval:
                    maxeigval = eigenvalue
        # print np.real(maxeigval)
        return np.real(maxeigval)

    def estimate_init_matrix_max_eigval(self):
        w = 0.0
        for i in range(self.n):
            w += self.calculate_init_ci_index(i)
        w0 = sum(self.G.degree())
        return np.power(w / w0, 1.0 / (2 * self.unite_order))

    def estimate_matrix_max_eigval(self):
        w = 0.0
        for i in range(self.n):
            if self.influence_index[i] == 1:
                w += self.caculate_ci_index(i)
        return np.power(w/self.w0, 1.0/(2*self.unite_order))

    def caculate_w0(self):
        w0 = 0
        for edge in self.G.edges(data=True):
            w0 += np.power(edge[2]['diffusivity'], 2)
        return w0

    @staticmethod
    def paint_spectrum(eigenvalues):
        plt.figure()
        for eigenvalue in eigenvalues:
            plt.scatter(np.real(eigenvalue), np.imag(eigenvalue), c='b', s=30)
        plt.show()

    def save_influencer(self, unite_order, eig, index):
        eig = np.array(eig)
        result = np.array(index)
        result_path = '../influencer_data/%s_influencer_%d.txt' % (self.G_name, unite_order)
        eig_path = '../influencer_data/%s_eig_%d.txt' % (self.G_name, unite_order)
        np.savetxt(result_path, result, fmt='%d')
        np.savetxt(eig_path, eig, fmt='%f')


if __name__ == "__main__":
    # ci = CI('../graph_data/netscience_n1589_p2742.gpickle', 1)
    ci = CI('../graph_data/er_graph_n50_p0.2.gpickle', 3)
    print('maxeigval=', ci.max_eigval)
    print('maxiniteig=', ci.max_init_eigval)
    print('threshold=', ci.influence_threshold)
    ci.ci_algorithm()
    # ci.paint_spectrum(ci.caculate_matrix_eigenvalue(ci.create_non_backtracking_matrix()))
    # print ci.caculate_matrix_max_eigval(ci.caculate_matrix_eigenvalue(ci.create_non_backtracking_matrix()))
