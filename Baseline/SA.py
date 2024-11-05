from random import*
import numpy as np
from math import*
from matplotlib import pyplot as plt
import pandas as pd
import time


start=time.perf_counter()
class Vrp_Env():
    choicei = 1#np.random.randint(81, 101)
    # 读取节点文件(节点id及其相应位置id)
    # (13,3)
    node_sets = pd.read_csv(f"./datasets/node{choicei}.csv").values
    # 读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    # (25，5)
    ticoor_sets = pd.read_csv(f"./datasets/ticoor{choicei}.csv").values
    # 读取订单文件(送货节点、收货节点、货物量)
    # (9,4)
    order_sets = pd.read_csv(f"./datasets/order{choicei}.csv").values
    # print(order_sets)
    # print(ticoor_sets[node_sets[0][0]][3])
    # print(ticoor_sets[node_sets[0][0]][4])
    def __init__(self):
        #网点数量
        self.depot_num= 1
        #货主数量
        self.huozhu_num = 10
        #客户数量
        self.customer=30
        # 订单数量
        self.order_num=self.order_sets.shape[0]
        #订单下标id
        self.order_index=self.order_sets[:,0]
        #时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        #当前订单是否可以访问,初始化为1都可以访问
        # self.mask=np.repeat(1, self.order_sets.shape[0])
        # 系统总时间
        self.total_time = 0  # 大于24h
        # # 当前等待时间
        # self.wait_time = 0
        # 等待总时间
        self.wait_totalt = 0
        # 总行程
        self.total_len = 0
        #车辆最大载货体积
        # self.max_load=20
        #车辆行驶速度
        self.vehicle_speed=20.0

        # 系统当前时间
        self.cur_time = 0
        #车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])
        # 每个订单是否被处理(未处理为0)
        # self.order_status = np.repeat(0, self.order_num)
        # self.order_sta = np.repeat(0, self.order_num)
        #车辆当前负载
        self.cur_load = 0
env=Vrp_Env()
#随机初始化城市坐标
# number_of_citys = 100
# citys = []
# for i in range(number_of_citys):
#     citys.append([randint(1,100),randint(1,100)])
# citys = np.array(citys)

#由城市坐索引标计算距离矩阵
# distance = np.zeros((len(env.ticoor_index),len(env.ticoor_index)))
# for i in env.ticoor_index:
#     for j in env.ticoor_index:
#         distance[i][j] = sqrt((env.ticoor_sets[i][3]-env.ticoor_sets[j][3])**2+(env.ticoor_sets[i][4]-env.ticoor_sets[j][4])**2)
def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    # dist = sqrt((env.ticoor_sets[a][3]-env.ticoor_sets[b][3])**2+(env.ticoor_sets[a][4]-env.ticoor_sets[b][4])**2)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist

#初始化参数
iteration1 = 500               #外循环迭代次数
T0 = 10                      #初始温度，取大些
Tf = 1                           #截止温度，可以不用
alpha = 0.55                     #温度更新因子
iteration2 = 10                  #内循环迭代次数
fbest = 0                        #最佳距离

# 车辆最大载货体积
max_load = 20
#初始化初解
# x = []
# for i in range(100):
#     x.append(i)
# x=env.order_index
def InitialSol(index):
    x=index.copy()
    np.random.seed(0)
    np.random.shuffle(x)
    x = np.array(x)
    route = []
    while len(x):
        order_nos = []
        remained_cap = max_load
        i=0
        j=0
        for order_no in x:
            i+=1
            if remained_cap - env.order_sets[order_no][3]>= 0 and i-j==1:
                j+=1
                order_nos.append(order_no)
                route.append(env.order_sets[order_no][1])
                remained_cap = remained_cap - env.order_sets[order_no][3]
        song=order_nos.copy()
        np.random.shuffle(song)
        for order_no in song:
            route.append(env.order_sets[order_no][2])
        for i in range(len(order_nos)):
            x=np.delete(x,0)
    return route
        # break
initroute=InitialSol(env.order_index[:30] )
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

    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    depot_loc = np.array([node_locx, node_locy])
    distances = eucli(vehicle_loc, depot_loc)
    add_t = distances / env.vehicle_speed
    t = (total_time + add_t) % 24.0  # 可能到第二天才去执行订单
    cur_time = t
    total_time += add_t
    vehicle_loc = depot_loc
    total_len += distances
    nodes.append(0)

    return total_time,wait_totalt,total_len,nodes

btotal_times,bwait_times,btotal_lens,bnodes=distance(initroute)
nodebest = np.insert(initroute, 0, 0).copy()
xbest = bnodes.copy()
f_now = btotal_times
f_nowwait = bwait_times
f_nowlens= btotal_lens
xnode_now = nodebest.copy()
x_now=xbest.copy()
history_best=[]
history_best.append(f_now)
# for j in range(len(x) - 1):
#     fbest = fbest + distance[x[j]][x[j + 1]]
# fbest = fbest + distance[x[-1]][x[0]]
# xbest = x.copy()
# f_now = fbest
# x_now = xbest.copy()

for i in range(iteration1):
    for k in range(iteration2):
        #生成新解
        x = env.order_index.copy()[:30]
        np.random.shuffle(x)
        route = []
        while len(x):
            order_nos = []
            remained_cap = max_load
            i = 0
            j = 0
            for order_no in x:
                i += 1
                if remained_cap - env.order_sets[order_no][3] >= 0 and i - j == 1:
                    j += 1
                    order_nos.append(order_no)
                    route.append(env.order_sets[order_no][1])
                    remained_cap = remained_cap - env.order_sets[order_no][3]
            song = order_nos.copy()
            np.random.shuffle(song)
            for order_no in song:
                route.append(env.order_sets[order_no][2])
            for i in range(len(order_nos)):
                x = np.delete(x, 0)
        total_times, wait_times, total_lens, nodes = distance(route)
        #判断是否更新解
        if total_times <= f_now:
            f_now = total_times
            x_now = nodes.copy()
            f_nowwait = wait_times
            f_nowlens = total_lens
            xnode_now = np.insert(route, 0, 0)
        if total_times > f_now:
            deltaf = total_times - f_now
            if random() < exp(-deltaf/T0):
                f_now = total_times
                x_now = nodes.copy()
                f_nowwait = wait_times
                f_nowlens = total_lens
                xnode_now = np.insert(route, 0, 0)
        if total_times < btotal_times:
            btotal_times = total_times
            xbest = nodes.copy()
            bwait_times = wait_times
            btotal_lens = total_lens
            nodebest = np.insert(route, 0, 0)
        history_best.append(btotal_times)
        cur_end = time.perf_counter()
        print("运行耗时", cur_end - start)
        print("temperature：%s，local obj:%s best obj: %s" % (T0, total_times, btotal_times))
    T0 = alpha * T0                #更新温度

    # if T0 < Tf:                  #停止准则为最低温度时可以取消注释
    #     break
end=time.perf_counter()
print("运行耗时", end-start)
#打印最佳路线和最佳距离
print('最短路径：',xbest)
print('最短总时间：',btotal_times)
print('等待时间：',bwait_times)
print('总长度：',btotal_lens)


#绘制结果
plt.title('SA_TSP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(env.ticoor_sets[...,3],env.ticoor_sets[...,4],'ob',ms = 3)
xbests=[]
for i in xbest:
    xbests.append(np.argwhere(env.ticoor_index == i)[0][0])
aa=env.ticoor_sets[xbests[5]][4]
plt.plot(env.ticoor_sets[xbests,3],env.ticoor_sets[xbests,4])
# plt.plot([citys[xbest[-1],0],citys[xbest[0],0]],[citys[xbest[-1],1],citys[xbest[0],1]],ms = 2)
plt.show()
# 绘制收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.savefig('result1.png')
    plt.show()
plotObj(history_best)

