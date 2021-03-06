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


def Inc_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, n_cluster, sigma, N):
    Ra_size = nodes_stru + nodes_attri + 1
    again = True
    it = 0
    obj_func = -1
    extent = 0
    w_origin = w[:]
    delta_Ra = np.zeros((Ra_size, Ra_size))

    print("结构点 =", nodes_stru + 1, "   属性点 =", nodes_attri)
    # 第一次执行迭代之前，先计算出Ra，并选择出初始的聚类中心
    Ra_origin, Pal_list = rand_walk_distance_Ra(G, c_restart, L, nodes_stru, nodes_attri, m_attri, w, N)
    centers, cluster_result = centers_chosen(Ra_origin, nodes_stru, n_cluster, sigma)
    cluster_result_final = cluster_result[:]

    while again:
        Ra = Ra_origin + delta_Ra
        again = False
        for vi in range(Ra_size):
            maxCenter = cluster_result[vi, 0]
            maxDis = cluster_result[vi, 1]
            for cen in range(n_cluster):
                dis = Ra[vi, centers[cen]]
                if dis > maxDis and dis != 0:
                    if dis > 1:
                        print("dis =", dis)
                    maxDis = dis
                    maxCenter = cen
            cluster_result[vi, 0] = maxCenter
            cluster_result[vi, 1] = maxDis
        it += 1
        cluster_list = result_to_list(cluster_result)

        centers = centers_update(Ra, centers, cluster_result, n_cluster, nodes_stru, cluster_list)
        # 更新权值,记得判断-1的点
        w_new = weight_adjustment(G, cluster_list, centers, cluster_result, w, nodes_stru, nodes_attri, n_cluster)

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

        # print(obj_func)

        delta_w = [w_new[i] - w_origin[i] for i in range(nodes_attri+1)]
        w.clear()
        w = w_new[:]

        delta_Ra = Incremental_distance(delta_w, nodes_stru, nodes_attri, L, c_restart, Pal_list)

    path_res = './output/Inc-cluster_data/cluster_result_' + file_name + '.csv'
    df_res = pd.DataFrame(cluster_result_final)
    df_res.to_csv(path_res)

    return cluster_result_final


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
    for file in os.listdir(path_data):
        z = file.find("edges")
        if z > -1:
            start = time.process_time()
            file_name = file[:z-1]
            w, nodes_stru, nodes_attri, m_attri, G, N = load_graph(path_data+"/", file_name)

            # 未知聚类个数
            n_cluster = get_circle_num(path_data+"/", file_name)

            cluster_result = \
                Inc_cluster(G, c_restart, L, nodes_stru, nodes_attri, file_name, m_attri, w, n_cluster, sigma, N)
            nodes[file_name] = nodes_stru
            end = time.process_time()

            print("-----------------\nrun time:", end - start, "\n-----------------")

    exResult("Inc-cluster_data", path_data)