from SA_cluster import *


def Incremental_distance(delta_w, nodes_stru, nodes_attri, L, c, Pal_list):
    delta_Pv = np.zeros((nodes_stru + 1, nodes_stru + 1))
    delta_A = np.zeros((nodes_stru + 1, nodes_attri))
    delta_B = np.zeros((nodes_attri, nodes_stru + 1))
    delta_C = np.zeros((nodes_attri, nodes_attri))

    Pa1 = Pal_list[1]
    Pv1 = Pa1[:nodes_stru + 1, :nodes_stru + 1]
    A1 = Pa1[:nodes_stru + 1, -nodes_attri:]
    B1 = Pa1[-nodes_attri:, :nodes_stru + 1]

    for m in range(1, nodes_attri+1):
        Aai = A1[:, m - 1]
        Aai = Aai * delta_w[m]
        delta_A[:, m - 1] = Aai

    An_1 = A1 + delta_A

    delta_Pa_1 = np.hstack((delta_Pv, delta_A))
    delta_Pa_2 = np.hstack((delta_B, delta_C))
    delta_Pa = np.vstack((delta_Pa_1, delta_Pa_2))

    delta_Ra = c * (1 - c) * delta_Pa

    delta_Pv_old = delta_Pv
    delta_A_old = delta_A
    delta_B_old = delta_B
    delta_C_old = delta_C

    for l in range(2, L+1):
        delta_Pv = np.dot(delta_Pv_old, Pv1) + np.dot(delta_A_old, B1)
        delta_B = np.dot(delta_B_old, Pv1) + np.dot(delta_C_old, B1)

        Pal = Pal_list[l]
        Al = Pal[:nodes_stru + 1, -nodes_attri:]
        Cl = Pal[-nodes_attri:, -nodes_attri:]

        for m in range(1, nodes_attri+1):
            Aai = Al[:, m - 1]
            Aai = Aai * delta_w[m]
            delta_A[:, m - 1] = Aai

            Cai = Cl[:, m - 1]
            Cai = Cai * delta_w[m]
            delta_C[:, m - 1] = Cai

        delta_A = delta_A + np.dot(delta_Pv_old, An_1)
        delta_C = delta_C + np.dot(delta_B_old, An_1)

        delta_Pal_1 = np.hstack((delta_Pv, delta_A))
        delta_Pal_2 = np.hstack((delta_B, delta_C))
        delta_Pal = np.vstack((delta_Pal_1, delta_Pal_2))

        delta_Ra = delta_Ra + c * pow(1 - c, l) * delta_Pal

        delta_Pv_old = delta_Pv
        delta_A_old = delta_A
        delta_B_old = delta_B
        delta_C_old = delta_C

    return delta_Ra


def Inc_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, sigma, N, comScale, fuzzy_m, Epsilon, Eta, Gamma, non_exis):
    Ra_size = nodes_stru + nodes_attri + 1
    obj_func = -1
    it = 0
    x_centers = []
    nodes_real = nodes_stru + 1 - len(non_exis)
    w_origin = w[:]
    delta_Ra = np.zeros((Ra_size, Ra_size))
    print("结构点 = ", nodes_real, "   属性点 = ", nodes_attri)

    Ra_origin, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)
    centers, cluster_result = centers_chosen(Ra_origin, nodes_stru, nodes_real, sigma)
    for cen in centers:
        x_centers.append(Ra_origin[cen])

    while True:
        Ra = Ra_origin + delta_Ra
        U, centers, cluster_result = membershipMat(Ra, nodes_stru, x_centers, fuzzy_m, non_exis)
        print("n_cluster_pred = ", U.shape[1])

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
            if (np.abs(obj_func_new - obj_func) < Epsilon or it == len(centers) * 10) and len(centers) == len(
                    x_centers):
                cluster_result_overlap = partition_result_overlap(U, nodes_stru, centers, non_exis, Gamma)
                # print("n_cluster_pred = ", U.shape[1])
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

        print(obj_func)
        delta_w = [w_new[i] - w_origin[i] for i in range(nodes_attri + 1)]
        w.clear()
        w = w_new[:]

        delta_Ra = Incremental_distance(delta_w, nodes_stru, nodes_attri, L, c_restart, Pal_list)

    path_res = './output/Inc-cluster_data/cluster_result_' + file_name + '.csv'
    df_res = pd.DataFrame.from_dict(cluster_result_overlap, orient='index')
    df_res.to_csv(path_res)

    return cluster_result, cluster_result_overlap


if __name__=='__main__':
    m_attri = 0  # 属性数量
    w = []
    centers = []
    nodes = collections.defaultdict(int)
    # ----------自定义参数---------- #
    c_restart = 0.5  # 重启概率
    L = 5  # 随机游走步长
    sigma = 1  # 计算密度方程的参数
    per_numericalAttri = 5  # 数值类型数据划为同一类的相差比例
    fuzzy_m = 2  # 模糊c均值参数m
    Epsilon = 0.0001  # 目标函数停止误差
    Eta = 0.00025  # 隶属度最大最小差值的阈值
    comScale = 1  # 在取消中心的时候社区规模的最小值
    Gamma = 0.95  # 重叠社区划分的程度大小，越低则重叠程度越高
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
            file_name = file[:z - 1]
            w, nodes_stru, nodes_attri, m_attri, G, N, non_exis = \
                load_graph(path_data + "/", file_name, per_numericalAttri)

            n_cluster_real = get_circle_num(path_data + "/", file_name)  # 用于和估算的社区数量进行对比
            print("n_cluster_real = ", n_cluster_real)

            cluster_result, cluster_result_overlap = \
                Inc_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name,
                           m_attri, w, sigma, N, comScale, fuzzy_m, Epsilon, Eta, Gamma, non_exis)
            nodes[file_name] = nodes_stru
            end = time.process_time()

            print("-----------------\nrun time:", end - start, "\n-----------------", '\n\n')
            break
    exResult("Inc-cluster_data", path_data)