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
def centers_chosen(Ra, nodes_stru, n_cluster, sigma):
    density = []
    Ra_size = Ra.shape[0]
    cluster_result = np.zeros((Ra_size, 2))
    cluster_result[:, :] = -1

    for vi in range(nodes_stru+1):
        density.append(cal_density(Ra, sigma, vi, nodes_stru))

    density = np.array(density)
    index = heapq.nlargest(n_cluster, range(len(density)), density.take)
    centers = list(index)

    for i in range(n_cluster):
        cluster_result[centers[i], 0] = i  # 选出的中心点作为第一批聚类结果
        cluster_result[centers[i], 1] = 1  # dis = 1 代表概率

    return centers, cluster_result


def cal_density(Ra, sigma, vi, nodes_stru):
    f = 0
    for vj in range(nodes_stru+1):
        f += (1-exp(-(pow(Ra[vi][vj], 2))/(2*pow(sigma, 2))))
    return f


def centers_update(Ra, centers, cluster_result, n_cluster, nodes_stru, cluster_list):
    nodes = Ra.shape[0]
    Ra_ave = np.zeros((n_cluster, nodes))

    # 计算聚类内各点到所有的游走距离的平均值

    for cen in range(n_cluster):
        for vj in range(nodes):
            N_cen = len(cluster_list[cen])
            sum_Ra = builtins.sum(Ra[cluster_list[cen][i]][vj] for i in range(N_cen))
            Ra_ave[cen][vj] = sum_Ra/N_cen

    # vi为聚类内的某点，vi_ave为聚类内所有点的平均值
    for cluster in range(n_cluster):
        N_cen = len(cluster_list[cluster])
        dis_euc = inf
        Updated = False
        cen = centers[cluster]
        for i in range(N_cen):
            vi = cluster_list[cluster][i]
            vec_vi = Ra[vi]
            vec_vi_ave = Ra_ave[cluster]
            dis = np.linalg.norm(vec_vi_ave-vec_vi)
            if dis < dis_euc and vi <= nodes_stru and vi != cen:
                Updated = True
                dis_euc = dis
                centers[cluster] = vi

        if Updated:
            # 改变聚类中心点，中心为自身，且dis = 1
            vi = centers[cluster]
            cluster_result[vi, 1] = 1

            # 改变原来聚类中心
            cluster_result[cen, 1] = Ra[vi][cen]

    return centers


def objective_function(Ra, cluster_list, n_cluster):
    sum = 0
    for k in range(n_cluster):
        V = len(cluster_list[k])
        sum_k = 0

        for i in range(V):
            sum_k_j = builtins.sum(Ra[cluster_list[k][i]][cluster_list[k][j]] for j in range(V))
            sum_k += sum_k_j
        sum += sum_k / V * V

    return sum


def result_to_list(cluster_result):
    nodes = cluster_result.shape[0]
    cluster_list = collections.defaultdict(list)

    for i in range(nodes):
        cluster = cluster_result[i][0]
        if cluster != -1:
            cluster_list[cluster].append(i)

    return cluster_list


def w_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, n_cluster, sigma, N):
    Ra_size = nodes_stru + nodes_attri + 1
    again = True
    it = 0
    obj_func = -1
    extent = 0

    print("结构点 = ", nodes_stru + 1, "   属性点 = ", nodes_attri)
    # 第一次执行迭代之前，先计算出Ra，并选择出初始的聚类中心
    Ra, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)
    centers, cluster_result = centers_chosen(Ra, nodes_stru, n_cluster, sigma)
    cluster_result_final = cluster_result[:]

    while again:
        again = False
        for vi in range(Ra_size):
            maxCenter = cluster_result[vi, 0]
            maxDis = cluster_result[vi, 1]
            for cen in range(n_cluster):
                dis = Ra[vi, centers[cen]]
                if dis > maxDis and dis != 0:
                    maxDis = dis
                    maxCenter = cen
            cluster_result[vi, 0] = maxCenter
            cluster_result[vi, 1] = maxDis
        it += 1
        cluster_list = result_to_list(cluster_result)

        centers = centers_update(Ra, centers, cluster_result, n_cluster, nodes_stru, cluster_list)
        # 更新权值,记得判断-1的点

        obj_func_new = objective_function(Ra, cluster_list, n_cluster)

        again = True
        if obj_func_new > obj_func:
            obj_func = obj_func_new
            extent = 0
            cluster_result_final = cluster_result[:]
        else:
            extent += 1
            if extent == L + 1:
                print(obj_func)
                break

        Ra, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)

    path_res = './output/w-cluster_data/cluster_result_'+file_name+'.csv'
    df_res = pd.DataFrame(cluster_result_final)
    df_res.to_csv(path_res)

    return cluster_result_final


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


def load_graph(path, file_name):  # path是文件路径，name是文件名（不带后缀名）
    nodes_stru = 0  # 结构点的数量
    m_attri = 1  # 属性的个数
    G = nx.Graph()

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

    # 属性值属于第几类属性
    type_attri = []
    type_attri.append(0)
    for row in range(rows-1):
        type_attri.append(m_attri)
        if reader_list[row][1] != reader_list[row+1][1]:
            m_attri += 1
    type_attri.append(m_attri)
    nodes_attri = rows

    N = np.zeros(nodes_stru + nodes_attri + 1)
    for i in range(nodes_stru + 1):
        if G.has_node(i):
            # 结构点N[vi]
            N[i] = len(list(G.neighbors(i)))

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
                v_i = i+nodes_stru
                G.add_edge(v_i, index)
                G.add_edge(index, v_i)

    for i in range(nodes_stru + 1, nodes_stru + nodes_attri + 1):
        if G.has_node(i):
            N[i] = len(list(G.neighbors(i)))

    return w, nodes_stru, nodes_attri, m_attri, G, N


def get_circle_num(path, file_name):  # 根据文件已知聚类数量

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
    c_restart = 0.5     # 重启概率
    L = 5               # 随机游走步长
    n_cluster = 2       # 聚类个数
    sigma = 1           # 计算密度方程的参数
    # ----------自定义参数---------- #
    nodes.clear()

    path_test = "./data/test_data"
    path_facebook = "./data/facebook"
    # 选择不同的数据集
    path_data = path_facebook
    # for file in os.listdir(path_data):
    #     z = file.find("edges")
    #     if z > -1:
    #         start = time.process_time()
    #         file_name = file[:z-1]
    #         w, nodes_stru, nodes_attri, m_attri, G, N = load_graph(path_data+"/", file_name)
    #
    #         # 未知聚类个数
    #         n_cluster = get_circle_num(path_data+"/", file_name)
    #
    #         cluster_result = \
    #             w_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, n_cluster, sigma, N)
    #         nodes[file_name] = nodes_stru
    #         end = time.process_time()
    #
    #         print("-----------------\nrun time:", end-start, "\n-----------------", '\n\n')

    exResult("w-cluster_data", path_data)