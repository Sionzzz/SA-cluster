import pandas as pd
from numpy import *
import numpy as np
import csv
import os
import collections

from edge import edge
from modularity import cal_Q
from modularity import cal_Qds
from graph_helper import load_graph
from nmi_compute import NMI
from onmi import *
import networkx as nx
import modularitydensity
from modularitydensity import metrics


def Ex_mod(nodes_stru, path_file, file_num, path_data):
    mod = 0
    for file in os.listdir(path_file):
        result_list = list()
        result_set = collections.defaultdict(set)
        file_name = "cluster_result_" + file_num
        z = file.find(file_name)
        # 模块度数据处理
        if z > -1:
            result_set.clear()
            result_list.clear()

            path_result = path_file + "/" + file
            reader = csv.reader(open(path_result))

            for i, line in enumerate(reader):
                if i != 0 and int(float(line[1])) != -1:
                    vi = int(float(line[0]))
                    for j in range(1, len(line)):
                        if line[j] != '':
                            cluster = int(float(line[j]))
                            result_set[cluster].add(vi)
                if i > nodes_stru:
                    break

            for key in result_set.keys():
                result_list.append(result_set[key])

            path_list = path_file + "/" + "result_list_" + file_num + ".txt"
            s = str(result_list)
            s = s.replace("[", "[\n").replace("},", "},\n").replace("]", "\n]")

            file_list = open(path_list, 'w')
            file_list.write(s)
            file_list.close()

            # --------------------模块度不适应于属性网络------------------------------
            print("mod_" + file_num + ":")
            infile = path_data + "/" + file_num + ".edges"
            G = load_graph(infile)
            mod = cal_Q(result_list, G)
            print(mod)
            print("--------------------------------")
            # ----------------------------------------------------------------------
            return mod
    return -1


def Ex_mod_Qds(nodes_stru, path_file, file_num, path_data):
    for file in os.listdir(path_file):
        result_list = list()
        result_set = collections.defaultdict(set)
        file_name = "cluster_result_" + file_num
        z = file.find(file_name)
        # 模块度数据处理
        if z > -1:
            result_set.clear()
            result_list.clear()

            path_result = path_file + "/" + file
            reader = csv.reader(open(path_result))
            for i, line in enumerate(reader):
                if i != 0 and int(float(line[1])) != -1:
                    vi = int(float(line[0]))
                    for j in range(1, len(line)):
                        if line[j] != '':
                            cluster = int(float(line[j]))
                            result_set[cluster].add(vi)
                if i > nodes_stru:
                    break

            for key in result_set.keys():
                result_list.append(result_set[key])

            # --------------------模块度不适应于属性网络------------------------------
            print("mod_Qds_" + file_num + ":")
            infile = path_data + "/" + file_num + ".edges"
            G = load_graph(infile)
            mod_Qds = cal_Qds(result_list, G)
            print(mod_Qds)
            print("******************************************************")

            # ----------------------------------------------------------------------
            return mod_Qds
    return -1



from nmi_compute import NMI
def Ex_nmi(nodes_stru, path_file, file_num, path_data):
    # 先处理算法计算的结果
    keys1 = set()
    keys2 = set()
    labels_pred = []
    labels = []
    labels_pred_dict = collections.defaultdict(set)

    for file in os.listdir(path_file):
        file_name = "cluster_result_" + file_num
        z = file.find(file_name)
        if z > -1:
            path_result = path_file + "/" + file

            reader = csv.reader(open(path_result))
            for i, line in enumerate(reader):
                if i != 0 and int(float(line[1])) != -1:
                    vi = int(float(line[0]))
                    for j in range(1, len(line)):
                        if line[j] != '':
                            cluster = int(float(line[j]))
                            labels_pred_dict[vi].add(cluster)
                if i > nodes_stru:
                    break

            keys1 = set(labels_pred_dict.keys())
            break
    # test
    # print(labels_pred_dict)

    # 再处理circles文件中已知的分类
    labels_dict = collections.defaultdict(set)
    for file in os.listdir(path_data):
        file_name = file_num + ".circles"
        z = file.find(file_name)
        if z > -1:
            path_circles = path_data + "/" + file
            reader = csv.reader(open(path_circles), delimiter="\t")

            labels_dict.clear()
            for line in reader:
                index_end = len(line[0])
                circle_num = int(line[0][6:index_end])

                circle_size = len(line)
                for i in range(1, circle_size):
                    vi = int(line[i])
                    labels_dict[vi].add(circle_num)

            keys2 = set(labels_dict.keys())
            break
    # test
    # print(labels_dict)

    keys = keys1 & keys2
    key_list = sorted(keys)
    for i in range(len(key_list)):
        vi = key_list[i]
        labels_pred.append(labels_pred_dict[vi])
        labels.append(labels_dict[vi])

    # print(labels)
    # print(labels_pred)

    # list1 = []
    # list2 = []
    # for i in range(len(labels)):
    #     list1.append(list(labels[i])[0])
    #     list2.append(list(labels_pred[i])[0])
    # print(list1)
    # print(list2)

    nmi = onmi(labels, labels_pred, allNodes=None, variant="LFK")
    # nmi_1 = NMI(array(list1), array(list2))
    # print("nmi_1 =", nmi_1)
    print("nmi_" + file_num + ":")
    print(nmi)
    print("--------------------------------")

    return nmi


# import networkx as nx
# from modularitydensity.metrics import modularity_density
# from modularitydensity.fine_tuned_modularity_density import fine_tuned_clustering_qds
# def mod_Qds(nodes_stru, file_num):
#     infile = "C:/Users/zzz/Desktop/facebook/" + file_num + ".edges"
#     G = load_graph(infile)
#     adj = nx.to_scipy_sparse_matrix(G)
#     result_list = []
#     for file in os.listdir("C:/Users/zzz/Desktop/SA-cluster_data"):
#         file_name = "cluster_result_" + file_num
#         z = file.find(file_name)
#         if z > -1:
#             print(file)
#             path = "C:/Users/zzz/Desktop/SA-cluster_data/cluster_result_" + file_num + ".csv"
#             path_result = path + file
#
#             reader = csv.reader(open(path_result))
#             for i, line in enumerate(reader):
#                 if i != 0 and int(float(line[1])) != -1:
#                     cluster = int(float(line[1]))
#                     result_list.append(cluster)
#                 if i > nodes_stru:
#                     break
#     mod_Q = modularity_density(adj, result_list, np.unique(result_list))


def Ex_density(nodes_stru, path_file, file_num, path_data):
    infile = path_data + "/" + file_num + ".edges"
    print()
    G = load_graph(infile)
    result_set = collections.defaultdict(set)
    for file in os.listdir(path_file):
        file_name = "cluster_result_" + file_num
        z = file.find(file_name)
        if z > -1:
            print(file_name)
            path_result = path_file + "/" + file
            reader = csv.reader(open(path_result))
            for i, line in enumerate(reader):
                if i != 0 and int(float(line[1])) != -1:
                    vi = int(float(line[0]))
                    for j in range(1, len(line)):
                        if line[j] != '':
                            cluster = int(float(line[j]))
                            result_set[cluster].add(vi)
                if i > nodes_stru:
                    break

            sum = 0
            # print("result_list_keys")
            # print(result_list.keys())
            # print("result_list")
            # print(result_list)
            for cen in result_set.keys():
                for vi in result_set[cen]:
                    for vj in result_set[cen]:
                        if G.has_edge(vi, vj):
                            sum += 1

            sum = sum / 2.0
            density = float(sum) / len(G.edges())
            print("density_" + file_num + ":")
            print(density)
            print("--------------------------------")

            return density


# def Ex_entropy(ndoes_stru, path_file, file_num):



if __name__=='__main__':
    print("this is Ex_cal")
