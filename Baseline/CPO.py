import copy
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

startt = time.perf_counter()


class Vrp_Env():
    # 读取节点文件(节点id及其相应位置id)
    node_sets = pd.read_csv("datasets/node1.csv", dtype={'id': int, 'idt1': int, 'idt2': int}).values

    # 读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    ticoor_sets = pd.read_csv("datasets/ticoor1.csv", dtype={'id': int, 'start_time': float, 'end_time': float, 'coor_x': float,
                                                   'coor_y': float}).values

    # 读取订单文件(送货节点、收货节点、货物量)
    order_sets = pd.read_csv("datasets/order1.csv", dtype={'order_id': int, 'from_node_id': int, 'to_node_id': int,
                                                 'goods_volume': float}).values

    def __init__(self):
        # 网点数量
        self.depot_num = 1
        # 订单数量
        self.order_num = self.order_sets.shape[0]
        # 时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        # 车辆行驶速度
        self.vehicle_speed = 20.0
        # 将订单和节点数据转换为DataFrame格式
        self.orders_df = pd.DataFrame(self.order_sets,
                                      columns=["order_id", "from_node_id", "to_node_id", "goods_volume"])
        self.nodes_df = pd.DataFrame(self.node_sets, columns=["id", "idt1", "idt2"])


env = Vrp_Env()


def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist


# 解码函数，将随机键转换为节点索引序列
# 解码函数，将随机键转换为节点索引序列
def decode_solution(solution):
    order_status = [0] * env.order_num
    # 按照 solution 对节点进行排序
    sorted_indices = np.argsort(solution)
    # 遍历所有订单获取取送货ID
    warehouse_pairs = []
    arr = []
    for index in sorted_indices:
        order = env.orders_df.iloc[index]
        from_node_id = order['from_node_id']
        to_node_id = order['to_node_id']

        # 从节点数据中随机获取取货和送货的仓库ID (idt1 或 idt2)
        id_qu_options = env.nodes_df[env.nodes_df['id'] == from_node_id][['idt1', 'idt2']].values.flatten()
        id_song_options = env.nodes_df[env.nodes_df['id'] == to_node_id][['idt1', 'idt2']].values.flatten()

        # 转换为列表以确保可用性
        id_qu_options = id_qu_options.tolist()
        id_song_options = id_song_options.tolist()

        # 检查选项是否为空
        if not id_qu_options or not id_song_options:
            print(f"Warning: No options found for from_node_id: {from_node_id}, to_node_id: {to_node_id}")
            continue  # 跳过该订单

        # 随机选择id_qu和id_song
        id_qu = random.choice(id_qu_options)
        id_song = random.choice(id_song_options)
        arr.append(id_qu)
        arr.append(id_song)
        warehouse_pairs.append((id_qu, id_song))

    # 打乱所有的id_qu和id_song
    random.shuffle(arr)
    # print("仓库配对:", warehouse_pairs)
    # print("打乱后的数组:", arr)

    # 检查并交换
    for a, b in warehouse_pairs:
        # 找到当前打乱数组中A和B的位置
        if arr.index(a) > arr.index(b):
            # 交换A和B的位置
            index_a = arr.index(a)
            index_b = arr.index(b)
            arr[index_a], arr[index_b] = arr[index_b], arr[index_a]

    # 截去每个元素的最后一个数字并转换为数字
    route = [int(str(x)[:-1]) for x in arr if str(x)[:-1].isdigit()]  # 确保截取后是数字
    # 在数组前面加个0
    route.append(0)  # 将 0 插入到数组的开头
    return route


# 计算路径的总时间、等待时间、总距离和访问的节点
def distance(routs):
    # 这个node是节点
    # routs = np.insert(routs, 0, 0)  # 不加axis时，数据进行展开构成一维数组
    # 网点出发到下一个节点
    # node_idx = env.node_sets[routs[0]][1]
    nodes = []
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    vehicle_loc = np.array([node_locx, node_locy])
    nodes.append(0)
    cur_time = 0
    total_time = 0
    wait_totalt = 0
    total_len = 0
    for node in routs:
        # routs里第i个节点
        node_idx1 = env.node_sets[node][1]
        node_locx1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][3]
        node_locy1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][4]
        node_loc1 = np.array([node_locx1, node_locy1])
        distance1 = eucli(vehicle_loc, node_loc1)
        add_t1 = distance1 / env.vehicle_speed
        t1 = (total_time + add_t1) % 24.0  # 可能到第二天才去执行订单
        node_idx2 = env.node_sets[node][2]
        node_locx2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][3]
        node_locy2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][4]
        node_loc2 = np.array([node_locx2, node_locy2])
        distance2 = eucli(vehicle_loc, node_loc2)
        add_t2 = distance2 / env.vehicle_speed
        t2 = (total_time + add_t2) % 24.0
        # 如果取货节点位置没有变
        if (add_t1 == 0 or add_t2 == 0):
            cur_time = cur_time
            total_time = total_time
            vehicle_loc = vehicle_loc
            if cur_time <= 12:
                nodes.append(node_idx1)
            else:
                nodes.append(node_idx2)
        elif (env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][1] <= t1 <
              env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2]):
            # 符合第一个位置的时间窗，前往第一个位置
            cur_time = t1
            total_time += add_t1
            vehicle_loc = node_loc1
            total_len += distance1
            nodes.append(node_idx1)
        elif (t1 >= env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2] and
              env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1] <= t2 <=
              env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][2]):
            # 符合第二个位置的时间窗，前往第二个位置
            cur_time = t2
            total_time += add_t2
            vehicle_loc = node_loc2
            total_len += distance2
            nodes.append(node_idx2)
        elif (t1 >= env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2] and t2 <
              env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1]):
            # 超过第一个位置的时间窗，小于第二个位置的时间窗，在第二个位置等待访问
            wait_time = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1] - t2
            wait_totalt += wait_time
            cur_time = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1]
            total_time += add_t2 + wait_time
            vehicle_loc = node_loc2
            total_len += distance2
            nodes.append(node_idx2)
        elif (t2 > env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1]):
            # 超过第一、二个位置的时间窗，等到第二天在第第一个位置进行访问,以时间窗为准
            wait_time = 24.0 - t1
            wait_totalt += wait_time
            cur_time = 0
            total_time += add_t1 + wait_time
            vehicle_loc = node_loc1
            total_len += distance1
            nodes.append(node_idx1)

    return total_time, wait_totalt, total_len, nodes


# 目标函数
def objective_function(solution):
    total_time, wait_time, total_len, nodes = distance(solution)
    return total_time


# CPO算法
def CPO(search_agents, max_iterations, lowerbound, upperbound, dimensions, objective):
    # 初始化
    lowerbound = np.zeros(dimensions)
    upperbound = np.ones(dimensions)

    n_min = max(5, int(0.5 * search_agents))  # 最小种群规模
    T = 2  # 用于调整种群的循环数量
    alpha = 0.2  # 收敛率
    Tf = 0.8  # 防御机制权衡

    # 初始化种群和适应度
    X = np.random.uniform(lowerbound, upperbound, (search_agents, dimensions))
    fitness = np.zeros(search_agents)
    CPO_curve = np.zeros(max_iterations)

    # 计算初始适应度
    for i in range(search_agents):
        permutation = decode_solution(X[i, :])
        fitness[i] = objective(permutation)

    score = np.min(fitness)
    index = np.argmin(fitness)
    best_pos = X[index, :]
    best_route = decode_solution(best_pos)
    total_time, wait_time, total_len, nodes = distance(best_route)

    Xp = X.copy()
    t = 0

    iteration_times = []

    while t < max_iterations:
        iter_start_time = time.perf_counter()
        # print("迭代次数:", t)
        r2 = np.random.rand()
        for i in range(search_agents):
            U1 = np.random.rand(dimensions) > np.random.rand()

            if np.random.rand() < np.random.rand():
                if np.random.rand() < np.random.rand():
                    rand_index = np.random.randint(search_agents)
                    y = (X[i, :] + X[rand_index, :]) / 2
                    X[i, :] += np.random.randn(dimensions) * np.abs(2 * np.random.rand() * best_pos - y)
                else:
                    rand_index1 = np.random.randint(search_agents)
                    rand_index2 = np.random.randint(search_agents)
                    y = (X[i, :] + X[rand_index1, :]) / 2
                    X[i, :] = U1 * X[i, :] + (1 - U1) * (
                            y + np.random.rand() * (X[rand_index1, :] - X[rand_index2, :]))
            else:
                Yt = 2 * np.random.rand() * (1 - t / max_iterations) ** (t / max_iterations)
                U2 = np.random.rand(dimensions) < 0.5 * 2 - 1
                S = np.random.rand(dimensions)
                if np.random.rand() < Tf:
                    St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    S *= Yt * St
                    rand_index1 = np.random.randint(search_agents)
                    rand_index2 = np.random.randint(search_agents)
                    rand_index3 = np.random.randint(search_agents)
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (
                            X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index3, :])) - S
                else:
                    Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    vt = X[i, :]
                    rand_index = np.random.randint(search_agents)
                    Vtp = X[rand_index, :]
                    Ft = np.random.rand(dimensions) * (Mt * (-vt + Vtp))
                    S *= Yt * Ft
                    X[i, :] = (best_pos + (alpha * (1 - r2) + r2) * (U2 * best_pos - X[i, :])) - S

            # 边界检查
            X[i, :] = np.clip(X[i, :], lowerbound, upperbound)

            # 转换为排列并解码
            permutation = decode_solution(X[i, :])

            # 无需可行性检查，约束条件在 distance 函数中处理

            new_fitness = objective(permutation)

            if new_fitness < fitness[i]:
                Xp[i, :] = X[i, :]
                fitness[i] = new_fitness
                if new_fitness < score:
                    best_pos = X[i, :]
                    score = new_fitness
                    best_route = decode_solution(best_pos)
                    total_time, wait_time, total_len, nodes = distance(best_route)
            else:
                X[i, :] = Xp[i, :]

        CPO_curve[t] = score
        t += 1

        # 调整种群规模
        new_search_agents = int(n_min + (search_agents - n_min) * (
                1 - (t % (max_iterations // T)) / (max_iterations // T)))
        if new_search_agents < search_agents:
            sorted_indexes = np.argsort(fitness)
            X = X[sorted_indexes[:new_search_agents], :]
            Xp = Xp[sorted_indexes[:new_search_agents], :]
            fitness = fitness[sorted_indexes[:new_search_agents]]
            search_agents = new_search_agents

        iter_end_time = time.perf_counter()
        iter_time = iter_end_time - iter_start_time
        iteration_times.append(iter_time)
        print(f"最短时间per {score}")
        print(f"运行耗时per {iter_time}")
        print(f"{t}")

    return score, best_route, total_time, wait_time, total_len, nodes, CPO_curve, iteration_times


# 主程序
if __name__ == "__main__":
    # 运行CPO算法
    search_agents = 30  # 搜索代理的数量（种群规模）
    max_iterations = 50  # 最大迭代次数，与 ACO 代码一致

    score, best_route, total_time, wait_time, total_len, nodes, curve, iteration_times = CPO(
        search_agents=search_agents,
        max_iterations=max_iterations,
        lowerbound=0,
        upperbound=1,
        dimensions=env.order_sets.shape[0],
        objective=objective_function
    )

    end = time.perf_counter()

    print("运行耗时", end - startt)
    print('最短总时间：', score)
    print('最短路径：', nodes)
    print('最短路径长度：', len(nodes))
    print('等待时间：', wait_time)
    print('总长度：', total_len)

    # 绘制结果
    plt.title('CPO_VRP')
    plt.xlabel('x')
    plt.ylabel('y')

    # 绘制节点
    for idx in range(len(env.ticoor_sets)):
        x = env.ticoor_sets[idx][3]
        y = env.ticoor_sets[idx][4]
        plt.plot(x, y, 'ob', ms=3)
        plt.text(x, y, str(int(env.ticoor_sets[idx][0])))

    # 绘制路径
    route_nodes = []
    start_ticoor_id = env.node_sets[0][1]
    indices = np.argwhere(env.ticoor_index == start_ticoor_id)
    if indices.size == 0:
        raise ValueError(f"起始节点 ID {start_ticoor_id} 未在 ticoor_index 中找到。")
    vehicle_locx = env.ticoor_sets[indices[0][0]][3]
    vehicle_locy = env.ticoor_sets[indices[0][0]][4]
    route_nodes.append([vehicle_locx, vehicle_locy])  # 起点

    for node_id in nodes[1:]:
        indices = np.argwhere(env.ticoor_index == node_id)
        if indices.size == 0:
            print(f"错误：节点 ID {node_id} 未在 ticoor_sets 中找到。")
            continue
        node_data = env.ticoor_sets[indices[0][0]]
        x = node_data[3]
        y = node_data[4]
        route_nodes.append([x, y])

    route_nodes = np.array(route_nodes)
    plt.plot(route_nodes[:, 0], route_nodes[:, 1], '-r')

    plt.show()
