import copy
from math import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
import time
startt=time.perf_counter()
# citys = [[12,26],[55,28],[76,81],[82,13],[42,64],[58,35],[84,92],[17,51],[87,34],[92,83],[67,63],[77,88],[24,24],[59,68],[43,10],[64,21],[19,17],[41,98],[91,44],[98,67],[25,29],[99,50],[23,94],[4,61],[20,32],[66,77],[13,57],[97,37],[57,33],[62,9],[22,85],[38,70],[37,96],[44,100],[35,11],[18,86],[33,58],[27,47],[83,27],[79,5],[80,65],[88,20],[49,56],[30,41],[89,16],[15,46],[14,74],[53,71],[93,38],[74,55],[60,97],[51,12],[40,49],[86,6],[72,66],[11,80],[5,54],[81,52],[31,73],[8,89],[95,91],[90,42],[34,79],[28,4],[47,43],[69,40],[85,53],[50,69],[3,76],[21,95],[94,31],[65,72],[78,93],[46,19],[63,1],[9,30],[100,48],[26,3],[52,18],[1,36],[10,59],[48,75],[68,62],[54,87],[16,22],[36,45],[61,78],[75,82],[7,84],[96,14],[73,2],[39,23],[2,15],[29,99],[6,90],[70,25],[45,39],[32,8],[71,7],[56,60]]
class Vrp_Env():
    choicei = 1#random.randint(81, 101)
    #读取节点文件(节点id及其相应位置id)
    #(13,3)
    node_sets=pd.read_csv(f"./datasets/node{choicei}.csv").values
    # print(node_sets)
    # print(node_sets.shape)

    #读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    #(25，5)
    ticoor_sets = pd.read_csv(f"./datasets/ticoor{choicei}.csv").values
    # print(ticoor_sets[2][3])

    #读取订单文件(送货节点、收货节点、货物量)
    #(9,4)
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
    # dist = sqrt((env.ticoor_sets[a][3]-env.ticoor_sets[b][3])**2+(env.ticoor_sets[a][4]-env.ticoor_sets[b][4])**2)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist
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
n1=1+env.huozhu_num+env.customer


#初始化参数
m=30   #蚂蚁数量
dnum=env.order_num-1
n=2*env.order_num-2    #路线中节点数量
alpha=1           #信息素重要程度因子
beta=2             #启发函数重要程度因子
rho=0.1             #信息素挥发因子
Q=1                #常系数
Tau=np.ones([n,n])       #信息素矩阵
Table=np.zeros([m,n])    #路径记录表 m个蚂蚁走过的路径
iter=0             #迭代次数的初值
iter_max=40     #最大迭代次数
Route_best=np.zeros([iter_max,n+2])  #到该代为止最优的路线
Totimes_best=np.zeros([iter_max,1]) #到该代为止最小的路径距离
Totimes_ave=np.zeros([iter_max,1])  #各代路径的平均长度
Length_best=np.zeros([iter_max,1])
Watimes_best=np.zeros([iter_max,1])
# aa=Watimes_best[0][0]
Table=Table.astype(np.int)
Route_best=Route_best.astype(np.int)
# state=np.zeros([m,m],dtype=np.int) #订单状态
max_load=20
cur_load=np.zeros([m,1])*max_load

def upmask(self, state):
    self.mask = np.random.randint(1, 2, size=(state.shape[0], state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(state.shape[2]):
            if state[i][0][j] == 0:
                if self.order_sets[j][3] > self.max_load - self.cur_load:
                    self.mask[i][j] = 0
            if state[i][0][j] == 2:
                self.mask[i][j] = 0
    return self.mask

#迭代巡展最佳路径
while iter<iter_max:
    #随机产生各个蚂蚁的起点城市
    state = np.zeros([m, env.order_num], dtype=np.int)
    start=np.zeros([m,1])
    temp = np.random.choice([i for i in range(env.order_num)], m)  # randperm(n)
    for i in range(m):
        start[i]=env.order_sets[temp[i]][1]
        cur_load[i]=env.order_sets[temp[i]][3]
        state[i][temp[i]]=1
    Table[:,0]=start[:,0]  #蚂蚁初始城市位置

    citys_index=np.arange(dnum+1,n1)  #城市索引，从1到n
    #逐个蚂蚁的路径选择
    for i in range(m):
        #对于一个蚂蚁的逐个城市的选择
        for j in range(1,n):
            allow=[]
            mask=np.zeros([dnum,1],dtype=np.int)
            tabu=Table[i,0:j]  #已经访问过的城市集合
            # allow_index=~ismember(citys_index,tabu) #没有访问过的城市置1
            # allow=citys_index(allow_index)   #待访问的城市集合
            # for y in tabu:
            for h in range(dnum):
                if state[i][h]==0:
                    if env.order_sets[h][3] <= max_load - cur_load[i]:
                        allow.append(env.order_sets[h][1])
                        mask[h] = 1
                if state[i][h]==1:
                    allow.append(env.order_sets[h][2])
                    mask[h] = 1
            P=np.array(allow)*1.0
            #计算城市间转移概率
            for k in range(len(allow)):    #k=1:length(allow):
                P[k]=pow(Tau[int(tabu[-1]),int(allow[k])],alpha)*beta
            P=P/sum(P)
            #轮盘赌选择下一个访问城市
            Pc=np.cumsum(P)
            # target_index=find(Pc>=random)
            s=np.random.random()
            for k in range(len(Pc)):
                if Pc[k] >= s:
                    target_index = k
                    break
            #判断是第几个订单
            count=-1
            for o in range(dnum):
                if mask[o]==1:
                    count+=1
                if count == target_index:
                    order_no =o
                    break
            # for h in state[i]:
            #     if h==0 or h==1:
            #         count+=1
            #     if count==target_index:
            #         order_no=count
            #         break
            # target=allow[target_index]#下标
            # if target
            if state[i][order_no]==1:
                state[i][order_no] = 2
                cur_load[i]-=env.order_sets[order_no][3]
                Table[i, j] = env.order_sets[order_no][2] #记录下来，加入Table，成为下一个访问的城市
            if state[i][order_no]==0:
                state[i][order_no] = 1
                cur_load[i]+=env.order_sets[order_no][3]
                Table[i, j] = env.order_sets[order_no][1] #记录下来，加入Table，成为下一个访问的城市

    #计算各个蚂蚁的路径距离
    total_times = np.zeros([m, 1])
    wait_times = np.zeros([m, 1])
    Length=np.zeros([m,1])
    nodes = np.zeros([m, n+2])
    for i in range(m):  #i=1:m:
        Route=Table[i,:] #每个蚂蚁的路径
        Route=Route.astype(np.int)
        # for j in range(n-1):
        #     Length[i]+=D[Route[j],Route[j+1]]
        # Length[i]+=D[Route[n-1],Route[0]]
        total_times[i], wait_times[i], Length[i], nodes[i,:] = distance(Route)
        if total_times[i]==67.46933491:
            print('总时间：',total_times[i])
            print('路径：',nodes[i,:])
    #计算最短路径距离及平均距离
    if iter==0:
        min_totimes=np.min(total_times)
        min_index=np.argwhere(total_times==min_totimes)[0][0]
        aa=Totimes_best[iter]
        aaa=Totimes_best[iter][0]
        Totimes_best[iter]=min_totimes
        Totimes_ave[iter]= np.mean(total_times)
        Watimes_best[iter] = wait_times[min_index]
        Length_best[iter] = Length[min_index]
        Route_best[iter,:]=nodes[min_index,:]
    else:
        min_totimes=np.min(total_times)
        min_index=np.argwhere(total_times==min_totimes)[0][0]
        Totimes_best[iter]=np.min([Totimes_best[iter-1],min_totimes])
        Totimes_ave[iter]= np.mean(total_times)
        if Totimes_best[iter]==min_totimes:
            Route_best[iter,:]=nodes[min_index,:]
            Watimes_best[iter] = wait_times[min_index]
            Length_best[iter] = Length[min_index]
        else:
            Route_best[iter,:]=Route_best[(iter-1),:]
            Watimes_best[iter] = Watimes_best[(iter-1)]
            Length_best[iter] = Length_best[(iter-1)]

    #更新信息素
    Delta_Tau=np.zeros([n,n])
    #逐个蚂蚁计算
    for i in range(m):
        #逐个城市计算
        for j in range(n-1):
            Delta_Tau[int(Table[i,j]),int(Table[i,j+1])]+=Q/Length[i]
        Delta_Tau[int(Table[i,n-1]),int(Table[i,0])]+=Q/Length[i]
    Tau=(1-rho)*Tau+Delta_Tau

    endt = time.perf_counter()
    print("最短时间per", Totimes_best[iter])
    print("运行耗时per", endt - startt)
    #迭代次数加1，清空路径记录表
    iter=iter+1
    print(iter)
    Table=np.zeros([m,n])
# 结果显示
# [Shortest_Length,index]=min(Length_best);
Shortest_Totimes = np.min(Totimes_best)
index = np.argwhere(Totimes_best==Shortest_Totimes)[0][0]
Shortest_Route=Route_best[index,:]
Shortest_Watimes = Watimes_best[index]
Shortest_Length = Length_best[index]
end=time.perf_counter()
print("运行耗时", end-startt)
print('最短总时间：',Shortest_Totimes)
print('最短路径：',Shortest_Route)
print('等待时间：',Shortest_Watimes)
print('总长度：',Shortest_Length)

# disp(['最短距离：',num2str(Shortest_Length)])
# disp(['最短路径：',num2str([Shortest_Route Shortest_Route(1)])]);
# %绘图
# figure(1)
# plot([citys(Shortest_Route,1);citys(Shortest_Route(1),1)],[citys(Shortest_Route,2);citys(Shortest_Route(1),2)],'o-')
# xlabel('x'),ylabel('y'),title('the best route');
# figure(2)
# plot(1:iter_max,Length_best,'b')
# xlabel('The iteration number'),ylabel('the shortest length'),title('The best length in every iteration');
# figure(3)
# plot(1:iter_max,Length_ave,'r'),xlabel('The iteration number'),ylabel('the average length'),title('The average length in every iteration');
# x=citys[0][0]
#绘制结果
# citys=np.array(citys)
# plt.title('ACO_TSP')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.plot(env.ticoor_sets[...,3],env.ticoor_sets[...,4],'ob',ms = 3)
# # xbests=[]
# # for i in Shortest_Route:
# #     xbests.append(np.argwhere(env.ticoor_index == i)[0][0])
# plt.plot(citys[...,0],citys[...,1],'ob',ms = 3)
# plt.plot(citys[Shortest_Route,0],citys[Shortest_Route,1])
# plt.plot([citys[Shortest_Route[-1],0],citys[Shortest_Route[0],0]],[citys[Shortest_Route[-1],1],citys[Shortest_Route[0],1]],ms = 2)
# plt.show()
#绘制结果
plt.title('ACO_VRP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(env.ticoor_sets[...,3],env.ticoor_sets[...,4],'ob',ms = 3)
xbests=[]
for i in Shortest_Route:
    xbests.append(np.argwhere(env.ticoor_index == i)[0][0])
# aa=env.ticoor_sets[xbests[5]][4]
plt.plot(env.ticoor_sets[xbests,3],env.ticoor_sets[xbests,4])
# plt.plot([citys[xbest[-1],0],citys[xbest[0],0]],[citys[xbest[-1],1],citys[xbest[0],1]],ms = 2)
plt.show()
