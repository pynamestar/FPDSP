import copy
from math import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
import time
import torch
import torch.nn as nn
startt=time.perf_counter()
# citys = [[12,26],[55,28],[76,81],[82,13],[42,64],[58,35],[84,92],[17,51],[87,34],[92,83],[67,63],[77,88],[24,24],[59,68],[43,10],[64,21],[19,17],[41,98],[91,44],[98,67],[25,29],[99,50],[23,94],[4,61],[20,32],[66,77],[13,57],[97,37],[57,33],[62,9],[22,85],[38,70],[37,96],[44,100],[35,11],[18,86],[33,58],[27,47],[83,27],[79,5],[80,65],[88,20],[49,56],[30,41],[89,16],[15,46],[14,74],[53,71],[93,38],[74,55],[60,97],[51,12],[40,49],[86,6],[72,66],[11,80],[5,54],[81,52],[31,73],[8,89],[95,91],[90,42],[34,79],[28,4],[47,43],[69,40],[85,53],[50,69],[3,76],[21,95],[94,31],[65,72],[78,93],[46,19],[63,1],[9,30],[100,48],[26,3],[52,18],[1,36],[10,59],[48,75],[68,62],[54,87],[16,22],[36,45],[61,78],[75,82],[7,84],[96,14],[73,2],[39,23],[2,15],[29,99],[6,90],[70,25],[45,39],[32,8],[71,7],[56,60]]
class Vrp_Env():
    #读取节点文件(节点id及其相应位置id)
    #(13,3)
    node_sets=pd.read_csv("node10.csv").values
    # print(node_sets)
    # print(node_sets.shape)

    #读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    #(25，5)
    ticoor_sets = pd.read_csv("ticoor10.csv").values
    # print(ticoor_sets[2][3])

    #读取订单文件(送货节点、收货节点、货物量)
    #(9,4)
    order_sets = pd.read_csv("order10.csv").values
    # print(order_sets)
    # print(ticoor_sets[node_sets[0][0]][3])
    # print(ticoor_sets[node_sets[0][0]][4])
    def __init__(self):
        #网点数量
        self.depot_num= 1
        #货主数量
        self.huozhu_num = 40
        #客户数量
        self.customer=120
        # 订单数量
        self.order_num=self.order_sets.shape[0]
        #订单下标id
        self.order_index=self.order_sets[:,0]
        #时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        #车辆最大载货体积
        # self.max_load=20
        #车辆行驶速度
        self.vehicle_speed=20.0
        #车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])
env=Vrp_Env()

def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist

n1=1+env.huozhu_num+env.customer
D=np.zeros([n1,n1])
node_loc=np.zeros([n1,4])
# 当前节点所有仓库与其他所有节点得所有仓库距离均值
for i in range(n1):
    for j in range(n1):
        if i!=j:
            node_idx10 = env.node_sets[i][1]
            node_locx10 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx10)[0][0]][3]
            node_locy10 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx10)[0][0]][4]
            node_loc10 = np.array([node_locx10, node_locy10])
            node_idx11 = env.node_sets[i][2]
            node_locx11 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx11)[0][0]][3]
            node_locy11 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx11)[0][0]][4]
            node_loc11 = np.array([node_locx11, node_locy11])

            node_idx20 = env.node_sets[j][1]
            node_locx20 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx20)[0][0]][3]
            node_locy20 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx20)[0][0]][4]
            node_loc20 = np.array([node_locx20, node_locy20])
            node_idx21 = env.node_sets[j][2]
            node_locx21 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx21)[0][0]][3]
            node_locy21 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx21)[0][0]][4]
            node_loc21 = np.array([node_locx21, node_locy21])

            distance1 = eucli(node_loc10, node_loc20)
            distance2 = eucli(node_loc10, node_loc21)
            distance3 = eucli(node_loc11, node_loc20)
            distance4 = eucli(node_loc11, node_loc21)
            D[i,j]= (distance1+distance2+distance3+distance4)/4
        else:
            D[i,j]=1e-4

#计算城市间的相互距离
def distance(routs):
    #这个node是节点
    # routs = np.insert(routs, 0, 0)  # 不加axis时，数据进行展开构成一维数组
    #网点出发到下一个节点
    # node_idx = env.node_sets[routs[0]][1]
    nodes=[]
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    vehicle_loc = np.array([node_locx, node_locy])
    nodes.append(0)
    cur_time = 0
    total_time = 0
    wait_totalt=0
    total_len=0
    for node in routs:
        #routs里第i个节点
        node_idx1 = env.node_sets[node][1]
        node_locx1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][3]
        node_locy1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][4]
        node_loc1 = np.array([node_locx1, node_locy1])
        distance1=eucli(vehicle_loc, node_loc1)
        add_t1 = distance1 / env.vehicle_speed
        t1 = (total_time + add_t1) % 24.0  # 可能到第二天才去执行订单
        node_idx2 = env.node_sets[node][2]
        node_locx2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][3]
        node_locy2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][4]
        node_loc2 = np.array([node_locx2, node_locy2])
        distance2 =eucli(vehicle_loc, node_loc2)
        add_t2 =  distance2/ env.vehicle_speed
        t2 = (total_time + add_t2) % 24.0
        # 如果取货节点位置没有变
        if (add_t1 == 0 or add_t2 == 0):
            cur_time = cur_time
            total_time = total_time
            vehicle_loc = vehicle_loc
            if cur_time<=12:
                nodes.append(node_idx1)
            else:
                nodes.append(node_idx2)
        elif (env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][1] <= t1 <env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2]):
            # 符合第一个位置的时间窗，前往第一个位置
            cur_time = t1
            total_time += add_t1
            vehicle_loc = node_loc1
            total_len += distance1
            nodes.append(node_idx1)
        elif (t1 >= env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2] and env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1] <= t2 <=env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][2]):
            # 符合第二个位置的时间窗，前往第二个位置
            cur_time = t2
            total_time += add_t2
            vehicle_loc = node_loc2
            total_len += distance2
            nodes.append(node_idx2)
        elif (t1 >= env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][2] and t2 <env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][1]):
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

    return total_time,wait_totalt,total_len,nodes

#D初始化为所有节点距离，两个节点所有可能取均值
n1=1+env.huozhu_num+env.customer

iter_max = 150#85
n = 2 * env.order_num
veh_num = 1
Route_best = np.zeros([iter_max, n + veh_num * 2])
Totimes_best = np.zeros([iter_max, 1])
Tomu_best = np.zeros([iter_max, 1])
Length_best = np.zeros([iter_max, 1])
Watimes_best = np.zeros([iter_max, 1])
Vtotime_best = np.zeros([iter_max, veh_num])
Vwatime_best = np.zeros([iter_max, veh_num])
Route_best = Route_best.astype(np.int64)

def GWO(Route_best, Totimes_best, Length_best, Watimes_best, Tomu_best):
    # 初始化参数
    population_size = 30  # 群体规模（狼的数量）
    max_iterations = iter_max  # 最大迭代次数
    n_orders = env.order_num
    # n_nodes = n_orders * 2
    # veh_num = 8  # 车辆数量

    # 初始化狼群（解的集合）
    wolves = []  # 存储狼群（解）
    flags_nodes = []  # 存储每个狼对应的车辆分配

    # 初始化狼群
    for i in range(population_size):
        state = np.zeros(n_orders, dtype=int)  # 订单状态，0：未访问，1：取货完成，2：送货完成
        routes = []  # 每辆车的路径
        flags_node = []  # 节点对应的车辆标记
        assigned_orders = np.arange(n_orders)
        np.random.shuffle(assigned_orders)
        orders_per_vehicle = n_orders // veh_num
        remainder = n_orders % veh_num
        idx = 0
        for v in range(veh_num):
            route = []
            num_orders = orders_per_vehicle + (1 if v < remainder else 0)
            vehicle_orders = assigned_orders[idx:idx+num_orders]
            idx += num_orders
            for order_idx in vehicle_orders:
                route.append(env.order_sets[order_idx][1])  # 取货点
                route.append(env.order_sets[order_idx][2])  # 送货点
                flags_node.extend([v, v])
            routes.extend(route)
        wolves.append(routes)
        flags_nodes.append(flags_node)

    # 主循环
    for iter in range(max_iterations):
        total_times = np.zeros(population_size)
        wait_times = np.zeros(population_size)
        Length = np.zeros(population_size)
        nodes_list = []
        maxve_time = np.zeros(population_size)
        vtotal_time = np.zeros((population_size, veh_num))
        vwait_time = np.zeros((population_size, veh_num))
        fitness = np.zeros(population_size)

        for i in range(population_size):
            Route = wolves[i]
            Flag_node = flags_nodes[i]
            total_times[i], wait_times[i], Length[i], nodes = distance(Route)
            mu = total_times[i]
            fitness[i] = mu
            nodes_list.append(nodes)

        # 找到alpha、beta、delta狼
        sorted_indices = np.argsort(fitness)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]

        alpha_wolf = wolves[alpha_idx]
        beta_wolf = wolves[beta_idx]
        delta_wolf = wolves[delta_idx]
        alpha_flags = flags_nodes[alpha_idx]
        beta_flags = flags_nodes[beta_idx]
        delta_flags = flags_nodes[delta_idx]

        # 更新狼的位置
        a = 2 - iter * (2 / max_iterations)  # a从2线性减小到0

        for i in range(population_size):
            if i in [alpha_idx, beta_idx, delta_idx]:
                continue
            X = wolves[i]
            X_flags = flags_nodes[i]
            X_new = []
            X_new_flags = []
            for pos in range(len(X)):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 1.618 * a * r1 - a
                C1 = 0.01 * r2

                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 1.618 * a * r1 - a
                C2 = 0.01 * r2

                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 1.618 * a * r1 - a
                C3 = 0.01 * r2

                # 从alpha、beta、delta狼中选择节点
                D_alpha = abs(C1 * alpha_wolf[pos] - X[pos])
                D_beta = abs(C2 * beta_wolf[pos] - X[pos])
                D_delta = abs(C3 * delta_wolf[pos] - X[pos])
                X1 = alpha_wolf[pos] - A1 * D_alpha
                X2 = beta_wolf[pos] - A2 * D_beta
                X3 = delta_wolf[pos] - A3 * D_delta
                X_new_pos = (X1 + X2 + X3) / 3

                # 由于我们的节点是整数，需要取整并处理越界情况
                X_new_pos = int(round(X_new_pos))
                if X_new_pos < 0:
                    X_new_pos = 0
                elif X_new_pos >= n1:
                    X_new_pos = n1 - 1

                X_new.append(X_new_pos)
                # 对应的车辆标记也进行更新
                X_new_flags.append(X_flags[pos])

            # 修复解，确保没有重复的节点，并满足取送货顺序
            X_new_fixed = []
            X_new_flags_fixed = []
            visited_pickups = set()
            visited_deliveries = set()
            for idx, node in enumerate(X_new):
                if node in env.order_sets[:,1]:  # 取货点
                    order_idx = np.where(env.order_sets[:,1] == node)[0][0]
                    if order_idx not in visited_pickups:
                        X_new_fixed.append(node)
                        X_new_flags_fixed.append(X_new_flags[idx])
                        visited_pickups.add(order_idx)
                elif node in env.order_sets[:,2]:  # 送货点
                    order_idx = np.where(env.order_sets[:,2] == node)[0][0]
                    if order_idx in visited_pickups and order_idx not in visited_deliveries:
                        X_new_fixed.append(node)
                        X_new_flags_fixed.append(X_new_flags[idx])
                        visited_deliveries.add(order_idx)
            # 补充遗漏的取送货点
            for order_idx in range(n_orders):
                if order_idx not in visited_pickups:
                    X_new_fixed.append(env.order_sets[order_idx][1])
                    X_new_flags_fixed.append(np.random.randint(0, veh_num))
                if order_idx not in visited_deliveries:
                    X_new_fixed.append(env.order_sets[order_idx][2])
                    X_new_flags_fixed.append(np.random.randint(0, veh_num))

            wolves[i] = X_new_fixed
            flags_nodes[i] = X_new_flags_fixed

        # 更新最佳解
        min_mu = np.min(fitness)
        min_index = np.argwhere(fitness == min_mu)[0][0]
        Tomu_best[iter] = min_mu
        Totimes_best[iter] = total_times[min_index]
        Watimes_best[iter] = wait_times[min_index]
        Length_best[iter] = Length[min_index]
        # 确保分配的数据维度一致
        min_len = min(len(Route_best[iter]), len(nodes_list[min_index]))  # 找到较小的长度进行操作

        # 如果 Route_best 的当前迭代行长度不足以容纳 nodes_list[min_index]，则扩展其大小
        if len(Route_best[iter]) < len(nodes_list[min_index]):
            # 动态扩展 Route_best 数组
            Route_best = np.resize(Route_best, (Route_best.shape[0], len(nodes_list[min_index])))
        Route_best[iter,:len(nodes_list[min_index])] = nodes_list[min_index]
        Vtotime_best[iter] = vtotal_time[min_index]
        Vwatime_best[iter] = vwait_time[min_index]

        endt = time.perf_counter()
        print("目标值", Tomu_best[iter])
        print("最短时间per", Totimes_best[iter])
        print('每辆车总时间：', Vtotime_best[iter])
        print('每辆车等待时间：', Vwatime_best[iter])
        print("运行耗时per", endt - startt)
        print(iter+1)

    return Totimes_best,Route_best,Watimes_best,Length_best,Vtotime_best,Vwatime_best,Tomu_best

# 结果显示
Totimes_best,Route_best,Watimes_best,Length_best,Vtotime_best,Vwatime_best,Tomu_best=GWO(Route_best,Totimes_best,Length_best,Watimes_best,Tomu_best)
Shortest_mu=np.min(Tomu_best)
index = np.argwhere(Tomu_best==Shortest_mu)[0][0]

Shortest_Totimes= Totimes_best[index]
Shortest_Route=Route_best[index,:]
Shortest_Watimes = Watimes_best[index]
Shortest_Length = Length_best[index]
Shortest_Vtotime=Vtotime_best[index]
Shortest_Vwatime=Vwatime_best[index]
end=time.perf_counter()
print("运行耗时CoTime", end-startt)
print("目标值", Shortest_mu)
print('最短总时间：',Shortest_Totimes)
print('最短路径：',Shortest_Route)
print('等待时间：',Shortest_Watimes)
print('总长度Length：',Shortest_Length)
print('时间AllTime：',Shortest_Vtotime)
print('等待时间WaitTime：',Shortest_Vwatime)

def extract_subarrays(arr):
    subarrays = []
    start_index = 0
    end_index = 0

    for i in range(len(arr)):
        if arr[i] == 0:
            if start_index != end_index:
                subarray = arr[start_index-1:end_index+1]
                subarrays.append(subarray)
            start_index = i + 1
            end_index = i + 1
        else:
            end_index += 1

    return subarrays

#绘制结果
plt.figure(figsize=(8,6),dpi=450)
plt.title('EGWO_VRP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(env.ticoor_sets[...,3],env.ticoor_sets[...,4],'ko',ms = 3)
xbests = [[] for _ in range(veh_num)]
subarrays = extract_subarrays(Shortest_Route)
for j, subarray in enumerate(subarrays):
    for i in subarray:
        xbests[j].append(np.argwhere(env.ticoor_index == i)[0][0])
# aa=env.ticoor_sets[xbests[5]][4]
colors = ['red', 'blue', 'green','purple','pink']
for k in range(veh_num):
    plt.plot(env.ticoor_sets[xbests[k],3],env.ticoor_sets[xbests[k],4],colors[k])
# plt.plot([citys[xbest[-1],0],citys[xbest[0],0]],[citys[xbest[-1],1],citys[xbest[0],1]],ms = 2)
plt.legend(['All Points','Route 1', 'Route 2', 'Route 3', 'Route 4', 'Route 5'])
plt.savefig('testblueline.jpg')
plt.show()
