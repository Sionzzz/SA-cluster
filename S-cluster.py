import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import functools
import networkx as nx
import os
import collections
import csv
import heapq


# 先计算转移概率矩阵Pa，再根据Pa求得Ra
def rand_walk_distance_Ra(G, c_restart, L, nodes_stru, file_name):
    # 概率转移矩阵Pa(受权值w影响)
    Pa = Matrix_transition(G, nodes_stru)
    df = pd.DataFrame(Pa)
    Pa_path = 'C:/Users/zzz/Desktop/S-cluster_data/Pa_' + file_name + '.csv'
    df.to_csv(Pa_path)

    size = nodes_stru+1
    Ra = np.zeros((size, size))

    Pal = np.identity(size)
    for l in range(1, L+1):
        Pal = np.dot(Pal, Pa)
        Ra += c_restart * pow(1 - c_restart, l) * Pal

    path_Ra = 'C:/Users/zzz/Desktop/S-cluster_data/Ra_'+file_name+'.csv'
    df_Ra = pd.DataFrame(Ra)
    df_Ra.to_csv(path_Ra)

    return Ra

def centers_chosen(Ra, n_cluster, sigma):  # 返回密度方程值最大的n_cluster个点的下标
    density = []
    Ra_size = Ra.shape[0]
    cluster_result = np.zeros((Ra_size, 2))
    cluster_result[:, :] = -1

    for vi in range(Ra_size):
        density.append(cal_density(Ra, sigma, vi, Ra_size))

    density = np.array(density)
    index = heapq.nlargest(n_cluster, range(len(density)), density.take)
    centers = list(index)

    for i in range(n_cluster):
        cluster_result[centers[i], 0] = i  # 选出的中心点作为第一批聚类结果
        cluster_result[centers[i], 1] = 1  # dis = 1 代表概率
        # print("cen = ", i, "  center = ", centers[i])
    return centers, cluster_result

def cal_density(Ra, sigma, vi, Ra_size):
    f = 0
    for vj in range(Ra_size):
        f += (1-exp(-(pow(Ra[vi][vj], 2))/(2*pow(sigma, 2))))
    return f

def centers_update(Ra, centers, cluster_result, n_cluster, file_name):
    nodes = Ra.shape[0]
    vi_cluster = collections.defaultdict(list)
    # 得到各个点属于哪一个聚类（聚类的编号）
    for vi in range(nodes):
        cen = int(cluster_result[vi, 0])
        if cen != -1:
            vi_cluster[cen].append(vi)
    Ra_ave = np.zeros((n_cluster, nodes))

    # 计算聚类内各点到所有的游走距离的平均值
    for cen in range(n_cluster):
        for vj in range(nodes):
            sum_Ra = 0
            N_cen = len(vi_cluster[cen])
            for i in range(N_cen):
                vi = vi_cluster[cen][i]
                sum_Ra += Ra[vi][vj]
            Ra_ave[cen][vj] = sum_Ra/N_cen
    path_Ra_ave = 'C:/Users/zzz/Desktop/S-cluster_data/Ra_ave_'+file_name+'.csv'
    df_Ra_ave = pd.DataFrame(Ra_ave)
    df_Ra_ave.to_csv(path_Ra_ave)

    # vi为聚类内的某点，vi_ave为聚类内所有点的平均值
    for cen in range(n_cluster):
        N_cen = len(vi_cluster[cen])
        dis_euc = inf
        for i in range(N_cen):
            vi = vi_cluster[cen][i]
            vec_vi = Ra[vi]
            vec_vi_ave = Ra_ave[cen]
            dis = np.linalg.norm(vec_vi_ave-vec_vi)
            if dis < dis_euc:
                dis_euc = dis
                centers[cen] = vi
    return centers


def weight_adjustment(vote_i, centers, w, nodes_stru, nodes_attri, n_cluster):
    # 分子-各个聚类中心点vote所有点（第i个属性）
    # 分母-同分子，是所有属性的加和
    vote_sum = 0
    vote_sum_i = []
    vote_sum_i.append(0)
    # print(centers)
    m = nodes_attri
    for i in range(1, m+1):  # m种属性
        sum_i = 0
        for k in range(n_cluster):  # n_cluster个聚类
            cen = centers[k]
            if cen <= nodes_stru:  # 中心点是结构点
                if cen in vote_i[i]:
                    sum_i += len(vote_i[i])
            else:  # 中心点是属性点
                sum_i += len(vote_i[i]) + 1
        vote_sum_i.append(sum_i)
        vote_sum += sum_i

    w_new = w[:]
    for i in range(1, m+1):
        delta_w = m * vote_sum_i[i] / vote_sum
        w_new[i] = 0.5 * (w_new[i] + delta_w)
    return w_new


def cal_vote(G, file_name, nodes_stru, nodes_attri):
    vote_i = collections.defaultdict(set)
    # 结构点-结构点
    for i in range(1, nodes_attri+1):
        for j in range(nodes_stru):
            for k in range(j+1, nodes_stru+1):
                vi = j
                vj = k
                attri = nodes_stru + i
                if G[vi][attri] == 1 and G[vj][attri] == 1:
                    vote_i[i].add(vi)
                    vote_i[i].add(vj)
    # print(vote_i[1])
    # print(len(vote_i[1]))
    return vote_i


def objective_function(Ra, cluster_list, n_cluster):
    nodes = Ra.shape[0]
    sum = 0

    for k in range(n_cluster):
        V = len(cluster_list[k])
        sum_k = 0
        for i in range(V):
            for j in range(V):
                vi = cluster_list[k][i]
                vj = cluster_list[k][j]
                sum_k += Ra[vi][vj]
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


def S_cluser(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, type_attri, n_cluster, sigma):
    Ra_size = nodes_stru + 1
    again = True

    print("结构点 = ", nodes_stru + 1, "   属性点 = ", nodes_attri)
    vote_i = cal_vote(G, file_name, nodes_stru, nodes_attri)
    # 第一次执行迭代之前，先计算出Ra，并选择出初始的聚类中心
    Ra = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, file_name)
    centers, cluster_result = centers_chosen(Ra, n_cluster, sigma)
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
                    again = True
            cluster_result[vi, 0] = maxCenter
            cluster_result[vi, 1] = maxDis

        centers = centers_update(Ra, centers, cluster_result, n_cluster, file_name)

    path_res = 'C:/Users/zzz/Desktop/S-cluster_data/cluster_result_'+file_name+'.csv'
    df_res = pd.DataFrame(cluster_result)
    df_res.to_csv(path_res)
    return cluster_result_final


# ----------------------------------------------------------------------------------------------------------------------
# 以上为S_cluster主要处理部分
# ----------------------------------------------------------------------------------------------------------------------
# 以下为数据读取处理,生成必要的数据
# ----------------------------------------------------------------------------------------------------------------------

def Matrix_transition(G, nodes_stru):
    size = nodes_stru+1
    Pa = np.zeros((size, size))
    N = np.zeros(size)
    # 计算点的邻居的数量数组N
    for i in range(size):
        for j in range(size):
            N[i] += G[i][j]
    # 生成转移概率矩阵Pa
    for i in range(size):
        for j in range(size):
            if G[i][j] == 1:
                Pa[i][j] = 1/N[i]
    return Pa


def load_graph(G, path, file_name):  # path是文件路径，name是文件名（不带后缀名）

    nodes_stru = 0  # 结构点的数量
    m_attri = 1  # 属性的个数
    graph = nx.Graph()
    G.clear()

    path_edges = path+file_name+".edges"
    reader = csv.reader(open(path_edges), delimiter=' ')  # edges两两数字组合，代表这两个序号点有边
    for line in reader:  # line是数字对，len（line) = 2
        v_i = int(line[0])
        v_j = int(line[1])
        G[v_i][v_j] = 1
        G[v_j][v_i] = 1  # 构造无向图

        Mnodes = max(v_j, v_i)
        nodes_stru = nodes_stru if nodes_stru > Mnodes else Mnodes
        graph.add_edge(v_i, v_j, color='b')

    # 计算属性种类数
    path_featnames = path+file_name+".featnames"
    reader = csv.reader(open(path_featnames), delimiter=' ')
    rows = len(open(path_featnames).readlines())  # 文件中的行数（不是从0开始）
    reader_list = list(reader)
    type_attri = []  # 属性值属于第几类属性
    type_attri.append(0)
    for row in range(rows-1):
        type_attri.append(m_attri)
        if reader_list[row][1] != reader_list[row+1][1]:
            m_attri += 1
    type_attri.append(m_attri)
    # for i in range(rows+1):
    #     print("i = ", i, "  type = ", type_attri[i])
    # print("属性种类数 = ", m_attri)
    nodes_attri = rows

    # 初始化权值
    w = []
    for i in range(nodes_attri+1):  # 初始化权重都为1
        w.append(1)

    # 生成属性扩展图
    path_feat = path+file_name+".feat"
    reader = csv.reader(open(path_feat), delimiter=' ')
    for line in reader:
        index = int(line[0])
        for i in range(1, 1+nodes_attri):
            num = int(line[i])
            if num != 0:
                v_i = i+nodes_stru
                G[v_i][index] = 1
                G[index][v_i] = 1
                # print("i = ", i, "  line[i] = ", line[i])

    # 直接去改，不确定是否有更合适的方法
    size = nodes_stru+nodes_attri+1
    for i in range(size):
        for j in range(size):
            if i not in G.keys():
                G[i][j] = 0
            else:
                if j not in G[i].keys():
                    G[i][j] = 0

    return w, nodes_stru, nodes_attri, m_attri, type_attri
    # 生成图形
    # nx.draw(graph, node_size=100, linewidths=1.0)
    # pic_name = file_name + "_graph.png"
    # plt.Svefig("C:/Users/zzz/Desktop/facebook_pic/" + pic_name)
    # plt.show()


def get_circle_num(file_name):
    path = "C:/Users/zzz/Desktop/facebook/"
    path_circles = path + file_name + ".circles"
    reader = csv.reader(open(path_circles), delimiter="\t")
    n = 0
    for line in reader:
        n += 1
    return n

if __name__=='__main__':
    m_attri = 0  # 属性数量
    w = []
    centers = []
    G = collections.defaultdict(collections.defaultdict)
    nodes = collections.defaultdict(int)
    # ----------自定义参数---------- #
    c_restart = 0.5  # 重启概率
    L = 5  # 随机游走步长
    n_cluster = 0  # 聚类个数
    sigma = 1  # 计算密度方程的参数
    # ----------自定义参数---------- #
    nodes.clear()
    for file in os.listdir("C:/Users/zzz/Desktop/facebook"):
        z = file.find("edges")
        if z > -1:
            file_name = file[:z-1]
            w, nodes_stru, nodes_attri, m_attri, type_attri = load_graph(G, "C:/Users/zzz/Desktop/facebook/", file_name)
            n_cluster = get_circle_num(file_name)
            cluster_result = \
                S_cluser(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, type_attri, n_cluster, sigma)
            nodes[file_name] = nodes_stru

            print(file_name)

    output_nodes = np.zeros((10, 2))
    nodes_dict = collections.defaultdict(int)
    index = -1
    for file in os.listdir("C:/Users/zzz/Desktop/facebook"):
        z = file.find("edges")
        if z > -1:
            index += 1
            file_name = file[:z-1]
            path_nodes = "C:/Users/zzz/Desktop/facebook/" + file_name + ".edges"
            reader = csv.reader(open(path_nodes), delimiter=' ')
            node_num = -1
            for line in reader:
                numA = int(line[0])
                numB = int(line[1])
                node_max = numA if numA > numB else numB
                node_num = node_max if node_max > node_num else node_num

            nodes_dict[file_name] = node_num
            output_nodes[index][0] = int(file_name)
            output_nodes[index][1] = node_num

    path_output_nodes = "C:/Users/zzz/Desktop/S-cluster_data/nodes.csv"
    df_nodes = pd.DataFrame(output_nodes)
    df_nodes.to_csv(path_output_nodes)


    from Ex_cal import Ex_mod, Ex_nmi
    mod_result = collections.defaultdict(int)
    mod_result_list = []
    nmi_result = collections.defaultdict(int)
    nmi_result_list = []
    for file in nodes_dict.keys():
        nodes_num = nodes_dict[file]

        mod = Ex_mod(nodes_num, file)
        mod_result[file] = mod
        mod_result_list.append(mod_result[file])

        nmi = Ex_nmi(nodes_num, file)
        nmi_result[file] = nmi
        nmi_result_list.append(nmi_result[file])

    path_mod = "C:/Users/zzz/Desktop/S-cluster_data/mod.csv"
    df_mod = pd.DataFrame(mod_result_list)
    df_mod.to_csv(path_mod)

    path_nmi = "C:/Users/zzz/Desktop/S-cluster_data/nmi.csv"
    df_nmi = pd.DataFrame(nmi_result_list)
    df_nmi.to_csv(path_nmi)