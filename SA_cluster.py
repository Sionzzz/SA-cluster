import pandas as pd
from numpy import *
import numpy as np
import networkx as nx
import os
import collections
import csv
import heapq
import time
import builtins


# 先计算转移概率矩阵Pa，再根据Pa求得Ra
def rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N):
    # 概率转移矩阵Pa(受权值w影响)
    Pa = Matrix_transition(G, nodes_stru, nodes_attri, m_attri, w, N)

    size = nodes_attri + nodes_stru + 1
    Ra = np.zeros((size, size))
    Pal = np.eye(size)
    Pal_list = list()
    Pal_list.append(Pal)

    for l in range(1, L+1):
        Pal = np.dot(Pal, Pa)
        Pal_list.append(Pal)
        Ra = Ra + c_restart * pow(1 - c_restart, l) * Pal

    return Ra, Pal_list


# 返回密度方程值最大的n_cluster个点的下标
def centers_chosen(Ra, nodes_stru, nodes_real, sigma, G):
    density = []
    centers = []
    cluster_result = []

    sum_density = 0
    max_density = 0
    for vi in range(nodes_stru + 1):
        value_density = cal_density(Ra, sigma, vi, nodes_stru, G)
        density.append(value_density)
        max_density = value_density if max_density < value_density else max_density
        sum_density += value_density
    ave_density = sum_density / nodes_real
    ave_density = (1.5*ave_density + 0.5*max_density) / 2.0

    [centers.append(i) for i in range(nodes_stru+1) if density[i] > ave_density]
    print("center_num = ", len(centers))

    # density = np.array(density)
    # index = heapq.nlargest(n_cluster, range(len(density)), density.take)
    # centers = list(index)

    cen = 0
    for i in range(nodes_stru+1):
        if i not in centers:
            cluster_result.append(-1)
        else:
            cluster_result.append(cen)
            cen += 1

    return centers, cluster_result


def cal_density(Ra, sigma, vi, nodes_stru, G):
    f = 0
    for vj in range(nodes_stru+1):
        if G.has_node(vi) and G.has_node(vj):
            f += (1-exp(-(pow(Ra[vi][vj], 2))/(2*pow(sigma, 2))))
    return f


def centers_update(Ra, U, fuzzy_m, nodes_stru, centers, Eta, comScale, cluster_list):
    x_centers = []
    n_cluster = len(centers)
    min_comSize = inf
    for j in range(n_cluster):
        sum_mole = 0
        sum_deno = 0
        flag = True
        comSize = len(cluster_list[j])
        min_comSize = comSize if min_comSize > comSize else min_comSize

        if comSize <= comScale:
            cen = centers[j]
            min_U = inf
            max_U = -1
            for c in range(n_cluster):
                if min_U > U[cen][c]:
                    min_U = U[cen][c]
                if max_U < U[cen][c]:
                    max_U = U[cen][c]
            if np.abs(max_U - min_U) < Eta:
                flag = False
        if flag:
            for i in range(nodes_stru+1):
                vec_vi = Ra[i]
                sum_mole += ((U[i][j] ** fuzzy_m) * vec_vi)
                sum_deno += (U[i][j] ** fuzzy_m)
            vec_center = sum_mole / sum_deno
            x_centers.append(vec_center)

    return x_centers, min_comSize


def weight_adjustment(G, cluster_list, centers, cluster_result, w, nodes_stru, nodes_attri):
    vote_sum = 0
    vote_i = []
    w_new = w[:]
    vote_i.append(0)

    for p in range(1, nodes_attri+1):
        attri_p = nodes_stru + p
        sum_i = 0
        for cen in centers:
            cluster = cluster_result[cen]
            for i in range(len(cluster_list[cluster])):
                v = cluster_list[cluster][i]
                if G.has_edge(attri_p, cen) and G.has_edge(attri_p, v):
                    sum_i += 1

        vote_i.append(sum_i)
        vote_sum += sum_i

    m = nodes_attri

    for i in range(1, nodes_attri+1):
        delta = (m * float(vote_i[i])) / vote_sum
        w_new[i] = 0.5 * (w[i] + delta)

    return w_new


def objective_function(Ra, U, centers, nodes_stru, fuzzy_m):
    sum_J = 0
    for i in range(nodes_stru+1):
        for j in range(len(centers)):
            vec_cen = Ra[centers[j]]
            vec_i = Ra[i]
            norm_Euc = np.linalg.norm(vec_cen - vec_i)
            sum_J += ((U[i][j] ** fuzzy_m) * (norm_Euc ** 2))
    return sum_J


def result_to_list(cluster_result):
    nodes = len(cluster_result)
    cluster_list = collections.defaultdict(list)

    for i in range(nodes):
        cluster = cluster_result[i]
        if cluster != -1:
            cluster_list[cluster].append(i)

    return cluster_list


def result_to_list_overlap(cluster_result_overlap, nodes_stru):
    cluster_list = collections.defaultdict(list)

    for i in range(nodes_stru+1):
        clusters = cluster_result_overlap[i]
        for cluster in clusters:
            if cluster != -1:
                cluster_list[cluster].append(i)

    return cluster_list


def membershipMat(Ra, nodes_stru, x_centers, fuzzy_m, non_exis):
    centers = []
    visited = []
    for i in range(nodes_stru+1):
        visited.append(0)
    n_cluster = len(x_centers)
    D = np.zeros((nodes_stru+1, n_cluster))
    U = np.zeros((nodes_stru+1, n_cluster))
    for i in range(nodes_stru+1):
        for j in range(n_cluster):
            if i in non_exis:
                D[i][j] = -1
            else:
                vec_vi = Ra[i]
                vec_vj = x_centers[j]
                D[i][j] = np.linalg.norm(vec_vi-vec_vj)

    for i in range(nodes_stru+1):
        if i in non_exis:
            continue
        for j in range(n_cluster):
            sum_U = 0
            flag = False
            cluster = -1
            for k in range(n_cluster):
                if D[i][k] == 0 or D[i][j] == 0:
                    flag = True
                    if D[i][k] == 0:
                        cluster = k
                    else:
                        cluster = j
                    break
                sum_U += (D[i][j] / D[i][k]) ** (2.0 / (fuzzy_m-1))

            if flag:
                for clu in range(n_cluster):
                    if clu != cluster:
                        U[i][clu] = 0
                    else:
                        U[i][clu] = 1
                break
            else:
                U[i][j] = 1.0 / sum_U

    for j in range(n_cluster):
        U_col = U[:, j]
        max_U = -1
        cen = 0
        for i in range(nodes_stru+1):
            if U_col[i] > max_U and visited[i] != 1:
                cen = i
                max_U = U_col[i]

        visited[cen] = 1
        centers.append(cen)

    cluster_result = partition_result(U, nodes_stru, centers, non_exis)

    # sum_list = []
    # for i in range(nodes_stru+1):
    #     num = np.sum(list(U[i, :]))
    #     if abs(num - 1) > 0.000000001:
    #         print("error", U[i, :])
    #     sum_list.append(num)
    # print(sum_list)

    return U, centers, cluster_result


def partition_result(U, nodes_stru, centers, non_exis):
    cluster_result = []

    for i in range(nodes_stru+1):
        if i in non_exis:
            cluster_result.append(-1)
        else:
            if i in centers:
                cluster = centers.index(i)
            else:
                U_row = U[i, :]
                cluster = list(U_row).index(max(U_row))
            cluster_result.append(cluster)

    return cluster_result


def partition_result_overlap(U, nodes_stru, centers, non_exis, Gamma):
    cluster_result_overlap = collections.defaultdict(set)
    n_cluster = U.shape[1]

    for i in range(nodes_stru+1):
        if i in non_exis:
            cluster_result_overlap[i].add(-1)
        else:
            if i in centers:
                cluster = centers.index(i)
                cluster_result_overlap[i].add(cluster)
            else:
                U_row = U[i, :]
                threshold_U = max(U_row) * Gamma
                for j in range(n_cluster):
                    if U[i][j] >= threshold_U:
                        cluster_result_overlap[i].add(j)

    return cluster_result_overlap


def SA_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, sigma, N, comScale, fuzzy_m, Epsilon, Eta, Gamma, non_exis):
    obj_func = -1
    it = 0
    x_centers = []
    nodes_real = nodes_stru + 1 - len(non_exis)
    print(nodes_stru)
    print("结构点 = ", nodes_real, "   属性点 = ", nodes_attri)

    Ra, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)
    centers, cluster_result = centers_chosen(Ra, nodes_stru, nodes_real, sigma, G)

    for cen in centers:
        x_centers.append(Ra[cen])

    while True:
        U, centers, cluster_result = membershipMat(Ra, nodes_stru, x_centers, fuzzy_m, non_exis)
        # print("n_cluster_pred = ", U.shape[1])

        cluster_list = result_to_list(cluster_result)
        x_centers, comScale = centers_update(Ra, U, fuzzy_m, nodes_stru, centers, Eta, comScale, cluster_list)
        it += 1

        w_new = weight_adjustment(G, cluster_list, centers, cluster_result, w, nodes_stru, nodes_attri)
        obj_func_new = objective_function(Ra, U, centers, nodes_stru, fuzzy_m)
        # print(obj_func_new)

        if len(centers) != len(x_centers):
            it = 0

        if obj_func == -1:
            obj_func = obj_func_new
        else:
            if (np.abs(obj_func_new - obj_func) < Epsilon or it == len(centers)*10) and len(centers) == len(x_centers):

                cluster_result_overlap = partition_result_overlap(U, nodes_stru, centers, non_exis, Gamma)
                print("n_cluster_pred = ", U.shape[1])
                # cluster_list_overlap = result_to_list_overlap(cluster_result_overlap, nodes_stru)
                # print(cluster_list_overlap)

                cluster_result = partition_result(U, nodes_stru, centers, non_exis)
                # cluster_list = result_to_list(cluster_result)
                # print(cluster_list)

                # for key in cluster_list.keys():
                #     print(len(cluster_list[key]))
                break
            else:
                obj_func = obj_func_new

        # print(obj_func)
        w.clear()
        w = w_new[:]

        Ra, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)

    path_res = './output/SA-cluster_data/cluster_result_'+file_name+'.csv'
    df_res = pd.DataFrame.from_dict(cluster_result_overlap, orient='index')
    df_res.to_csv(path_res)

    return cluster_result, cluster_result_overlap


# ----------------------------------------------------------------------------------------------------------------------
# 以上为SA_cluster主要处理部分
# ----------------------------------------------------------------------------------------------------------------------
# 以下为数据读取处理,生成必要的数据
# ----------------------------------------------------------------------------------------------------------------------


def Matrix_transition(G, nodes_stru, nodes_attri, m_attri, w, N):
    size = nodes_attri+nodes_stru+1
    Pa = np.zeros((size, size))

    # 生成转移概率矩阵Pa
    for i in range(size):
        w_sum = N[i] * w[0] + nodes_attri

        for j in range(size):
            if G.has_edge(i, j):
                if i <= nodes_stru:  # 矩阵的上半部分
                    if j <= nodes_stru:
                        Pa[i][j] = w[0] / w_sum  # 结构点-结构点
                    else:
                        Pa[i][j] = w[j - nodes_stru] / w_sum  # 结构点-属性点
                else:
                    if j <= nodes_stru:
                        Pa[i][j] = 1 / N[i]  # 属性点-结构点
    return Pa


def load_graph(path, file_name, per_numericalAttri):  # path是文件路径，name是文件名（不带后缀名）
    nodes_stru = 0  # 结构点的数量
    m_attri = 1  # 属性的个数
    G = nx.Graph()
    non_exis = []

    path_edges = path+file_name+".edges"
    reader = csv.reader(open(path_edges), delimiter=' ')  # edges两两数字组合，代表这两个序号点有边
    for line in reader:  # line是数字对，len（line) = 2
        vi = int(line[0])
        vj = int(line[1])
        G.add_edge(vi, vj)
        G.add_edge(vj, vi)

        Mnodes = max(vj, vi)
        nodes_stru = nodes_stru if nodes_stru > Mnodes else Mnodes

    # 计算属性种类数
    path_featnames = path+file_name+".featnames"
    reader = csv.reader(open(path_featnames), delimiter=' ')
    rows = len(open(path_featnames).readlines())  # 文件中的行数（不是从0开始）
    reader_list = list(reader)

    # # 属性值属于第几类属性
    # type_attri = []
    # type_attri.append(0)
    # for row in range(rows-1):
    #     type_attri.append(m_attri)
    #     if reader_list[row][1] != reader_list[row+1][1]:
    #         m_attri += 1
    # type_attri.append(m_attri)
    nodes_attri = rows

    N = np.zeros(nodes_stru + nodes_attri + 1)
    for i in range(nodes_stru + 1):
        if G.has_node(i):
            # 结构点N[vi]
            N[i] = len(list(G.neighbors(i)))
        else:
            non_exis.append(i)

    # 初始化权值
    w = []
    for i in range(nodes_attri+1):
        w.append(1.0)

    # 生成属性扩展图
    path_feat = path+file_name+".feat"
    reader = csv.reader(open(path_feat), delimiter=' ')
    for line in reader:
        index = int(line[0])
        for i in range(1, 1+nodes_attri):
            num = int(line[i])
            if num != 0:
                v_i = i + nodes_stru
                G.add_edge(v_i, index)
                G.add_edge(index, v_i)

    # # 改进点：处理数值型属性（暂定）
    # NodeAndValue = collections.defaultdict(list)
    # attri_type = [] # 表示当前属性的类别，主要有类别，有无以及数值（两大类）
    # nodes_numericalAttri = 0 # 数值类型属性的个数
    # nodes_typeAttri = 0 # 类别型属性的个数
    # path_feat =
    # reader = csv.reader(open(path_feat), delimiter=' ')
    # for line in reader:
    #     node = line[0]
    #     for i in range(1, 1+nodes_attri):
    #         value = line[i]
    #         if attri_type[i] == 0: # 处理类别型属性(有无)
    #             nodes_typeAttri += 1
    #             if value != 0:
    #                 v_j = i + nodes_stru
    #                 G.add_edge(v_j, node)
    #                 G.add_edge(node, v_j)
    #
    #         elif attri_type[i] == 1: # 处理数值型属性
    #             nodes_numericalAttri += 1
    #             NodeAndValue[i].append((node, value))
    #
    # nodes_attri = nodes_typeAttri
    # for i in range(1, 1+nodes_numericalAttri):
    #     nodeValue_list = NodeAndValue[i]
    #     nodeValue_list.sort(key=lambda x: x[1])
    #     interval = -inf
    #     for j in range(len(nodeValue_list)):
    #         node = nodeValue_list[j][0]
    #         value = nodeValue_list[j][1]
    #
    #         if value <= interval:
    #             v_j = nodes_stru + nodes_attri
    #             G.add_edge(node, v_j)
    #             G.add_edge(v_j, node)
    #         else:
    #             interval = value + value * per_numericalAttri
    #             nodes_attri += 1
    #             v_j = nodes_stru + nodes_attri
    #             G.add_edge(node, v_j)
    #             G.add_edge(v_j, node)

    for i in range(nodes_stru + 1, nodes_stru + nodes_attri + 1):
        if G.has_node(i):
            N[i] = len(list(G.neighbors(i)))

    return w, nodes_stru, nodes_attri, m_attri, G, N, non_exis


def get_circle_num(path, file_name):  # 根据文件求聚类数量

    path_circles = path + file_name + ".circles"
    reader = csv.reader(open(path_circles), delimiter="\t")
    n = 0
    for line in reader:
        n += 1
    return n


def exResult(path_algorithm, path_data):
    output_nodes = collections.defaultdict(collections.defaultdict)
    nodes_dict = collections.defaultdict(int)
    index = -1
    for file in os.listdir(path_data):
        z = file.find("edges")
        if z > -1:
            index += 1
            file_name = file[:z - 1]
            path_nodes = path_data + "/" + str(file_name) + ".edges"
            reader = csv.reader(open(path_nodes), delimiter=' ')
            node_num = -1
            for line in reader:
                numA = int(line[0])
                numB = int(line[1])
                node_max = numA if numA > numB else numB
                node_num = node_max if node_max > node_num else node_num

            nodes_dict[file_name] = node_num
            output_nodes[index][0] = str(file_name)
            output_nodes[index][1] = node_num

    path_output_nodes = "./output/" + path_algorithm + "/nodes.csv"
    df_nodes = pd.DataFrame(output_nodes)
    df_nodes.to_csv(path_output_nodes)

    from Ex_cal import Ex_mod, Ex_nmi, Ex_mod_Qds, Ex_density
    mod_result = collections.defaultdict(int)
    mod_result_list = []
    nmi_result = collections.defaultdict(int)
    nmi_result_list = []
    Qds_result = collections.defaultdict(int)
    Qds_result_list = []
    density_result = collections.defaultdict(int)
    density_result_list = []

    path_output = "./output/" + path_algorithm
    for file in nodes_dict.keys():
        nodes_num = nodes_dict[file]

        density = Ex_density(nodes_num, path_output, file, path_data)
        density_result[file] = density
        density_result_list.append(density_result[file])

        mod = Ex_mod(nodes_num, path_output, file, path_data)
        mod_result[file] = mod
        mod_result_list.append(mod_result[file])

        nmi = Ex_nmi(nodes_num, path_output, file, path_data)
        nmi_result[file] = nmi
        nmi_result_list.append(nmi_result[file])

        mod_Qds = Ex_mod_Qds(nodes_num, path_output, file, path_data)
        Qds_result[file] = mod_Qds
        Qds_result_list.append(Qds_result[file])

    path_mod = path_output + "/mod.csv"
    df_mod = pd.DataFrame(mod_result_list)
    df_mod.to_csv(path_mod)

    path_mod_Qds = path_output + "/mod_Qds.csv"
    df_mod_Qds = pd.DataFrame(mod_result_list)
    df_mod_Qds.to_csv(path_mod_Qds)

    path_nmi = path_output + "/nmi.csv"
    df_nmi = pd.DataFrame(nmi_result_list)
    df_nmi.to_csv(path_nmi)

    path_density = path_output + "/density.csv"
    df_density = pd.DataFrame(density_result_list)
    df_density.to_csv(path_density)


if __name__=='__main__':
    m_attri = 0  # 属性数量
    w = []
    centers = []
    nodes = collections.defaultdict(int)
    # ----------自定义参数---------- #
    c_restart = 0.5             # 重启概率
    L = 5                       # 随机游走步长
    sigma = 1                   # 计算密度方程的参数
    per_numericalAttri = 5      # 数值类型数据划为同一类的相差比例
    fuzzy_m = 2                 # 模糊c均值参数m
    comScale = 1                # 在取消中心的时候社区规模的最小值
    Epsilon = 0.0001            # 目标函数停止误差
    Eta = 0.0015                # 隶属度最大最小差值的阈值
    Gamma = 0.95                # 重叠社区划分的程度大小，越低则重叠程度越高
    # ----------自定义参数---------- #
    nodes.clear()

    path_test = "./data/test_data"
    path_facebook = "./data/facebook"
    # 选择不同的数据集
    path_data = path_facebook

    for file in os.listdir(path_data):
        z = file.find("edges")
        if z > -1:
            start = time.process_time()
            file_name = file[:z-1]
            w, nodes_stru, nodes_attri, m_attri, G, N, non_exis = \
                load_graph(path_data+"/", file_name, per_numericalAttri)

            n_cluster_real = get_circle_num(path_data+"/", file_name)   # 用于和估算的社区数量进行对比
            print("n_cluster_real = ", n_cluster_real)

            cluster_result, cluster_result_overlap = \
                SA_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name,
                           m_attri, w, sigma, N, comScale, fuzzy_m, Epsilon, Eta, Gamma, non_exis)
            nodes[file_name] = nodes_stru
            end = time.process_time()

            print("-----------------\nrun time:", end-start, "\n-----------------", '\n\n')

    exResult("SA-cluster_data", path_data)