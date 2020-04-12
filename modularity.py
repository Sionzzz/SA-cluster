def cal_Q(partition, G):
    m = len(G.edges(None, False))
    a = []
    e = []

    # 与第i个社区中的节点相连的边在所有边中占的比例
    for community in partition:
        t = 0.0
        for node in community:
            if G.has_node(node):
                t += len(list(G.neighbors(node)))
        a.append(t/(2*m))

    # 网络中社区内部节点之间相连的边数在网络总的边数中所占的比例
    for community in partition:
        # print(community)
        community = list(community)
        t = 0.0
        for i in range(len(list(community))):
            for j in range(len(list(community))):
                vi = community[i]
                vj = community[j]
                if G.has_node(vi) and G.has_node(vj):
                    if G.has_edge(vi, vj):
                        t += 1.0
        e.append(t/(2*m))

    # print("a = ", len(a))
    # print("e = ", len(e))
    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai**2) 
        
    return q


import collections
import numpy as np
def cal_Qds(partition, G):
    # partition list(set)
    # print("size = ", len(G.nodes()))
    # print(partition)
    E = len(G.edges(None, False))
    c = len(partition)

    v_com = collections.defaultdict(set)
    a_ic = collections.defaultdict(float)

    # community set
    for com, community in enumerate(partition):
        o_i = len(community)
        for v in community:
            vi = int(v)
            v_com[vi].add(com)

    for i in G:
        vi = int(i)
        # print(vi)
        # print(v_com[vi])
        if len(v_com[vi]) == 0:
            print(vi)
        a_ic[i] = 1.0 / len((v_com[vi]))

    Ec_in = []
    Ec_out = []
    dc = []
    dcc = np.zeros((c, c))
    Ecc = np.zeros((c, c))

    for com, community in enumerate(partition):
        sum_in = 0
        sum_out = 0
        sum_dc = 0
        for i in community:
            vi = int(i)
            for j in community:
                vj = int(j)
                if G.has_edge(vi, vj):
                    sum_in += a_ic[vi]
                if vi != vj:
                    sum_dc += a_ic[vi]

            for j in G:
                vj = int(j)
                com_vj = v_com[vj]
                if com not in com_vj:
                    f = (a_ic[vi] + a_ic[vj]) / 2.0
                    if G.has_edge(vi, vj):
                        sum_out += f
                        for com1 in com_vj:
                            Ecc[com][com1] += f
                    for com2 in com_vj:
                        dcc[com][com2] += f

        Ec_out.append(sum_out)
        Ec_in.append(0.5 * sum_in)
        if sum_in == 0:
            dc.append(0)
        else:
            dc.append(sum_in / sum_dc)

    for i in range(c):
        for j in range(c):
            if Ecc[i][j] == 0:
                dcc[i][j] = 0
            else:
                dcc[i][j] = Ecc[i][j] / dcc[i][j]

    # print(Ec_in)
    # print(Ec_out)
    # print(dc)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for com, community in enumerate(partition):
        sum1 += Ec_in[com] / E * dc[com]
        tmp2 = (2 * Ec_in[com] + Ec_out[com]) * dc[com] / (2 * E)
        tmp2 = tmp2 ** 2
        sum2 += tmp2

        sum3_1 = 0
        for com1, community1 in enumerate(partition):
            if com1 != com:
                sum3_1 += Ecc[com][com1] * dcc[com][com1]
        sum3 += sum3_1 / (2 * E)

        # print("sum1 = ", sum1, " sum2 = ", sum2, " sum3 = ", sum3)

    return sum1 - sum2 - sum3


if __name__ == "__main__":
    print("this is modularity")