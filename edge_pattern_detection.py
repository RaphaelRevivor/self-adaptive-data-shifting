import numpy as np
import numpy.linalg as lin
import csv
import copy
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

class Data_Shifting():
    def __init__(self, new_data):

        self.new_data = new_data
        self.num = self.new_data.shape[0]

        # 1. calculate the number of k
        self.k_num = int(round(5*np.log10(self.num)))
        # print (k_num)

        # 2. fit knn models
        clf = NearestNeighbors(n_neighbors=self.k_num, algorithm='ball_tree', metric = 'minkowski',p=2).fit(self.new_data)
        self.distances, indices = clf.kneighbors(self.new_data)
        neighbor_array = [[] for i in range(self.num)]

        # get nearest neighbors of all the data points
        for index, item in enumerate(indices):
            for sub_item in item:
                neighbor_array[index].append(self.new_data[sub_item])

        self.neighbor_array = np.array(neighbor_array)

    def outlier_generation(self):
        self.neighbor_array_sub = copy.deepcopy(self.neighbor_array)

        # 3. calculate x_i - x_ij
        for neighbors in self.neighbor_array_sub:
            temp = copy.deepcopy(neighbors[0])
            for index, item in enumerate(neighbors):
                neighbors[index] = temp - item

        # 4. calculate V_ij, V_ij = (x_i - x_ij)/||x_i - x_ij||
        neighbor_array_norm = copy.deepcopy(self.neighbor_array_sub)

        for i in range(self.num):
            for j in range(self.k_num-1):
                if self.distances[i,j] == 0:
                    # previously they are 1
                    neighbor_array_norm[i,j,0] = 0
                    neighbor_array_norm[i,j,1] = 0
                else:
                    neighbor_array_norm[i,j,0] = neighbor_array_norm[i,j,0] / self.distances[i,j]
                    neighbor_array_norm[i,j,1] = neighbor_array_norm[i,j,1] / self.distances[i,j]

        # 5. calculate the normal vector n_i, n_i = sum(V_ij)
        self.neighbor_array_sum = np.sum(neighbor_array_norm, axis = 1)
        edge_list = []
        cnt = 0
        # threshold for sum(theta_ij)
        threshold = 0.9

        # 6. calculate theta_ij = V_ij.T*n_i and , 
        # select the data point (indices) that exceeds the threshold as edge points
        for index,item in enumerate(self.neighbor_array_sub):
            cnt = 0.0
            for sub_item in item:
                if np.dot(sub_item.transpose(), self.neighbor_array_sum[index])>=0:
                    cnt += 1
            if cnt/self.k_num >= threshold:
                edge_list.append(index)

        # select corresponding data points as edge points
        self.edge_array = []
        for index in edge_list:
            self.edge_array.append(self.new_data[index,:])

        self.edge_array = np.array(self.edge_array)

        # 7. calculate l_ns and n_i/|n_i|
        neighbor_array_sum_sel = []
        l_ns = 0.0
        for index in edge_list:
            neigh_dist_sum = np.sum(self.distances[index])
            l_ns += neigh_dist_sum
            neighbor_array_sum_sel.append(self.neighbor_array_sum[index]/lin.norm(self.neighbor_array_sum[index]))

        # add a parameter here
        C = 1
        l_ns = l_ns / (C*len(edge_list)) /  self.k_num
        neighbor_array_sum_sel = np.array(neighbor_array_sum_sel)

        # 8. final artificial outlier array
        self.outlier_array =  self.edge_array + l_ns * neighbor_array_sum_sel

        return self.outlier_array

    def target_generation(self):
        # 9. calculate unit shifting direction array delta p(x_i)/||delta p(x_i)||
        # shifting direction should be the opposite of the outlier
        shift_dir_array = -self.neighbor_array_sum
        temp = lin.norm(shift_dir_array, axis=1).reshape(-1,1)
        unit_shift_dir = shift_dir_array / temp


        # 10. for each target data point, calculate the inner product, 
        # and select the smallest positive value as the positive shifting value
        self.pseudo_target_data = []
        inner_product_array = [[]for i in range(self.neighbor_array_sub.shape[0])]


        # 11. calculate the minimum projection distance
        for index,item in enumerate(self.neighbor_array_sub):
            min_projection = 1000.0
            for sub_item in item:
                temp_product = np.inner(sub_item.reshape(1,-1),unit_shift_dir[index].reshape(1,-1))
                temp_product = temp_product[0][0]
                if temp_product < min_projection and temp_product > 0:
                    min_projection = temp_product
            if min_projection!= 1000:
                inner_product_array[index].append(min_projection)
            else:
                inner_product_array[index].append(0)

        inner_product_array = np.array(inner_product_array)

        # 12. calculate the pseudo target data
        for index, item in enumerate(self.new_data):
            if inner_product_array[index][0] != 0.0:
                temp_pseudo = item + inner_product_array[index] * unit_shift_dir[index]
                self.pseudo_target_data.append(temp_pseudo)

        self.pseudo_target_data = np.array(self.pseudo_target_data)

        return self.pseudo_target_data

    def visualization(self):
        # 12'. plotting
        fig = plt.figure()
        ori = plt.scatter(self.new_data[:,0], self.new_data[:,1], c = 'k', lw=0)
        # plt.scatter(self.edge_array[:,0], self.edge_array[:,1], c = 'r', lw=0)
        out = plt.scatter(self.outlier_array[:,0], self.outlier_array[:,1], c = 'r', lw=0)
        tar = plt.scatter(self.pseudo_target_data[:,0], self.pseudo_target_data[:,1], c = 'g', lw=0)

        plt.legend((ori,out,tar),
           ('Target data', 'Artificial outliers', 'Pseudo target data'),
           loc='upper left',
           ncol=1,
           fontsize=12)
    
        plt.savefig('pseudo_outliers_targets.png')
        plt.show()