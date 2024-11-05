import copy

import numpy as np
import pandas as pd
import torch
import random

"""
    计算两个点之间的欧几里得距离
"""
# a=[]
def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist

class Vrp_Env():
    #读取节点文件(节点id及其相应位置id)
    #(13,3)
    # node_sets=pd.read_csv("node.csv").values
    # # print(node_sets)
    # # print(node_sets.shape)
    #
    # #读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    # #(25，5)
    # ticoor_sets = pd.read_csv("ticoor.csv").values
    # # print(ticoor_sets[2][3])
    #
    # #读取订单文件(送货节点、收货节点、货物量)
    # #(9,4)
    # order_sets = pd.read_csv("order.csv").values
    # print(order_sets)
    # print(ticoor_sets[node_sets[0][0]][3])
    # print(ticoor_sets[node_sets[0][0]][4])
    def __init__(self):
        self.node_sets = pd.read_csv("./datasets/node1.csv").values
        self.ticoor_sets = pd.read_csv("./datasets/ticoor1.csv").values
        self.order_sets = pd.read_csv("./datasets/order1.csv").values

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
        self.mask=np.repeat(1, self.order_sets.shape[0])
        # 系统总时间
        self.total_time = 0  # 大于24h
        # # 当前等待时间
        # self.wait_time = 0
        # 等待总时间
        self.wait_totalt = 0
        # 总行程
        self.total_len = 0
        #车辆最大载货体积
        self.max_load=20
        #车辆行驶速度
        self.vehicle_speed=20.0

        # 系统当前时间
        self.cur_time = 0
        #车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])
        # 每个订单是否被处理(未处理为0)
        self.order_status = np.repeat(0, self.order_num)
        self.order_status[self.order_num-1]=2
        self.order_sta = np.repeat(0, self.order_num)
        #车辆当前负载
        self.cur_load = 0
        # 车辆到每个订单当前目的地的距离,初始化为1
        mu = []  # [1,9,1,2,2,2,3,3,3]
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                mu.append(self.order_sets[i][1])
            if element == 1:
                mu.append(self.order_sets[i][2])
            if element == 2:
                mu.append(50)
        # ord_mu = sorted(mu)#[1,2,2,2,3,3,3,9,12]
        # only_mu = list(set(ord_mu)) #下一步可访问多有节点[1,2,3,9,12 ]
        self.dismu = np.ones((2, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for i, element in enumerate(mu):
            # print(i,element)
            if element != 50:
                element_idx1 = self.node_sets[element][1]
                element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                element_loc1 = np.array([element_locx1, element_locy1])
                element_t1 = eucli(self.vehicle_loc, element_loc1) / self.vehicle_speed
                self.dismu[0][i] = element_t1
                element_idx2 = self.node_sets[element][2]
                element_locx2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][3]
                element_locy2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][4]
                element_loc2 = np.array([element_locx2, element_locy2])
                element_t2 = eucli(self.vehicle_loc, element_loc2) / self.vehicle_speed
                self.dismu[1][i] = element_t2
            else:
                self.dismu[0][i] = 9
                self.dismu[1][i] = 9

        # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu = np.repeat(0, self.order_sets.shape[0])
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                if self.order_sets[i][3] <= self.max_load - self.cur_load:
                    self.locmu[i]= -self.order_sets[i][3]
            if element == 1:
                self.locmu[i]= self.order_sets[i][3]
            if element == 2:
                self.locmu[i] = 8
        # 全部订单是否全部完成
        self.done = False

        # 输入的状态是系统当前时间、车辆当前位置、订单的处理状态、车辆负载
        # self.state = (self.cur_time,self.vehicle_loc,self.order_status,self.cur_load)
        # 输入的状态是订单的处理状态
        # self.state = self.vehicle_loc,np.array([self.cur_load]),self.order_status
        # a.append(self.state)
        # print(self.state)
        # self.state_dim=self.state.shape[0]

    """
        环境重置
    """
    def reset(self):
        # 随机选择一个i值，范围从1到80
        choicei = random.randint(81, 100)#choicei = random.randint(1, 80)
        self.node_sets = pd.read_csv(f"./datasets/node{choicei}.csv").values
        self.ticoor_sets = pd.read_csv(f"./datasets/ticoor{choicei}.csv").values
        self.order_sets = pd.read_csv(f"./datasets/order{choicei}.csv").values

        # 当前订单是否可以访问,初始化为1都可以访问
        self.mask = np.repeat(1, self.order_sets.shape[0])
        # 车辆到每个订单当前目的地的距离,初始化为1
        # self.dismu = np.ones((2, self.order_sets.shape[0]))
        # # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        # self.locmu = np.repeat(0, self.order_sets.shape[0])
        # self.locmu =np.zeros((1, self.order_sets.shape[0]))
        # 系统总时间
        self.total_time = 0 #大于24h
        # # 当前等待时间
        # self.wait_time = 0
        # 等待总时间
        self.wait_totalt = 0
        # 总行程
        self.total_len = 0

        # 系统当前时间
        self.cur_time = 0 #一天之内的时间
        # 车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])
        # 每个订单是否被处理(未处理为0)
        self.order_status = np.repeat(0, self.order_num)
        self.order_status[self.order_num - 1] = 2
        self.order_sta = np.repeat(0, self.order_num)
        # 车辆当前负载
        self.cur_load = 0
        # 车辆到每个订单当前目的地的距离,初始化为1
        mu = []  # [1,9,1,2,2,2,3,3,3]
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                mu.append(self.order_sets[i][1])
            if element == 1:
                mu.append(self.order_sets[i][2])
            if element == 2:
                mu.append(50)
        # ord_mu = sorted(mu)#[1,2,2,2,3,3,3,9,12]
        # only_mu = list(set(ord_mu)) #下一步可访问多有节点[1,2,3,9,12 ]
        self.dismu = np.ones((2, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for i, element in enumerate(mu):
            # print(i,element)
            if element != 50:
                element_idx1 = self.node_sets[element][1]
                element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                element_loc1 = np.array([element_locx1, element_locy1])
                element_t1 = eucli(self.vehicle_loc, element_loc1) / self.vehicle_speed
                self.dismu[0][i] = element_t1
                element_idx2 = self.node_sets[element][2]
                element_locx2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][3]
                element_locy2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][4]
                element_loc2 = np.array([element_locx2, element_locy2])
                element_t2 = eucli(self.vehicle_loc, element_loc2) / self.vehicle_speed
                self.dismu[1][i] = element_t2
            else:
                self.dismu[0][i] = 9
                self.dismu[1][i] = 9

        # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu = np.repeat(0, self.order_sets.shape[0])
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                if self.order_sets[i][3] <= self.max_load - self.cur_load:
                    self.locmu[i] = -self.order_sets[i][3]
            if element == 1:
                self.locmu[i] = self.order_sets[i][3]
            if element == 2:
                self.locmu[i] = 8
        # 全部订单是否全部完成
        self.done = False

        # 输入的状态是系统当前时间、订单的处理状态、车辆当前位置、车辆负载
        # self.state = self.vehicle_loc,np.array([self.cur_load]),self.order_status
        return self.dismu,self.locmu.reshape(1,self.order_num),self.order_sta.reshape(1,self.order_num)


    """
        根据动作返回下一步的转态、奖励
    """
    def step(self,action):
        #传入的action是即将要处理的订单id
        #待送状态，去货主位置取货
        # print("self.order_status[action]",self.order_status[action])
        # distance = 0
        distance = 0
        if (self.order_status[action] == 0):
            node = self.order_sets[action][1]
            node_idx1 = self.node_sets[node][1]
            node_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][3]
            node_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][4]
            node_loc1 = np.array([node_locx1, node_locy1])
            distance1 = eucli(self.vehicle_loc, node_loc1)
            add_t1 = distance1 / self.vehicle_speed
            t1 = (self.total_time + add_t1) % 24.0  # 可能到第二天才去执行订单
            node_idx2 = self.node_sets[node][2]
            node_locx2 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][3]
            node_locy2 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][4]
            node_loc2 = np.array([node_locx2, node_locy2])
            distance2 = eucli(self.vehicle_loc, node_loc2)
            add_t2 = distance2 / self.vehicle_speed
            t2 = (self.total_time + add_t2) % 24.0
            # distance=distance1
            # 如果取货节点位置没有变
            if (add_t1 == 0 or add_t2 == 0):
                self.cur_time = self.cur_time
                self.total_time = self.total_time
                self.vehicle_loc = self.vehicle_loc
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load += self.order_sets[action][3]
                reward = 0
                if self.cur_time <= 12:
                    nodes = node_idx1
                else:
                    nodes = node_idx2
            elif (self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][1] <= t1 <
                  self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2]):
                # 符合第一个位置的时间窗，前往第一个位置
                self.cur_time = t1
                self.total_time += add_t1
                self.vehicle_loc = node_loc1
                # print("node_loc1",self.vehicle_loc)
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load += self.order_sets[action][3]
                reward = -add_t1
                distance = distance1
                nodes = node_idx1
            elif (t1 >= self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2] and
                  self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1] <= t2 <=
                  self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][2]):
                # 符合第二个位置的时间窗，前往第二个位置
                self.cur_time = t2
                self.total_time += add_t2
                self.vehicle_loc = node_loc2
                # print("node_loc2", self.vehicle_loc)
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load += self.order_sets[action][3]
                reward = -add_t2
                distance = distance2
                nodes = node_idx2
            elif (t1 >= self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2] and t2 <
                  self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]):
                # 超过第一个位置的时间窗，小于第二个位置的时间窗，在第二个位置等待访问
                wait_time = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1] - t2
                self.wait_totalt += wait_time
                self.cur_time = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]
                self.total_time += add_t2 + wait_time
                self.vehicle_loc = node_loc2
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load += self.order_sets[action][3]
                reward = -(add_t2 + wait_time)
                distance = distance2
                nodes = node_idx2
            elif (t2 > self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]):
                # 超过第一、二个位置的时间窗，等到第二天在第第一个位置进行访问,以时间窗为准
                wait_time = 24.0 - t1
                self.wait_totalt += wait_time
                self.cur_time = 0
                self.total_time += add_t1 + wait_time
                self.vehicle_loc = node_loc1
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load += self.order_sets[action][3]
                reward = -(add_t1 + wait_time)
                distance = distance1
                nodes = node_idx1
        # 在送状态，去客户位置送货
        elif (self.order_status[action] == 1):
            node = self.order_sets[action][2]
            if node==0:
                node_idx = self.node_sets[node][1]
                node_locx = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][3]
                node_locy = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][4]
                node_loc= np.array([node_locx, node_locy])
                distances = eucli(self.vehicle_loc, node_loc)
                add_t = distances / self.vehicle_speed
                t = (self.total_time + add_t) % 24.0  # 可能到第二天才去执行订单
                self.cur_time = t
                self.total_time += add_t
                self.vehicle_loc = node_loc
                self.order_status[action] = 10  # 返回网点任务完成
                self.order_sta = copy.deepcopy(self.order_status)
                self.cur_load -= self.order_sets[action][3]
                reward = -add_t
                distance = distances
                nodes = node_idx
            else:
                # node = self.order_sets[action][2]
                node_idx1 = self.node_sets[node][1]
                node_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][3]
                node_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][4]
                node_loc1 = np.array([node_locx1, node_locy1])
                distance1 = eucli(self.vehicle_loc, node_loc1)
                add_t1 = distance1 / self.vehicle_speed
                t1 = (self.total_time + add_t1) % 24.0  # 可能到第二天才去执行订单
                node_idx2 = self.node_sets[node][2]
                node_locx2 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][3]
                node_locy2 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][4]
                node_loc2 = np.array([node_locx2, node_locy2])
                distance2 = eucli(self.vehicle_loc, node_loc2)
                add_t2 = distance2 / self.vehicle_speed
                t2 = (self.total_time + add_t2) % 24.0
                # 如果送货节点位置没有变
                if (add_t1 == 0 or add_t2 == 0):
                    self.cur_time = self.cur_time
                    self.total_time = self.total_time
                    self.vehicle_loc = self.vehicle_loc
                    self.order_status[action] = 2  # 送货完成为完成状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    self.cur_load -= self.order_sets[action][3]
                    reward = 0
                    if self.cur_time <= 12:
                        nodes = node_idx1
                    else:
                        nodes = node_idx2
                elif (self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][1] <= t1 <
                      self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2]):
                    # 符合第一个位置的时间窗，前往第一个位置
                    self.cur_time = t1
                    self.total_time += add_t1
                    self.vehicle_loc = node_loc1
                    self.order_status[action] = 2  # 送货完成为完成状态态
                    self.order_sta = copy.deepcopy(self.order_status)
                    self.cur_load -= self.order_sets[action][3]
                    reward = -add_t1
                    distance = distance1
                    nodes = node_idx1
                elif (t1 >= self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2] and
                      self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1] <= t2 <=
                      self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][2]):
                    # 符合第二个位置的时间窗，前往第二个位置
                    self.cur_time = t2
                    self.total_time += add_t2
                    self.vehicle_loc = node_loc2
                    self.order_status[action] = 2  # 送货完成为完成状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    self.cur_load -= self.order_sets[action][3]
                    reward = -add_t2
                    distance = distance2
                    nodes = node_idx2
                elif (t1 >= self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2] and t2 <
                      self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]):
                    # 超过第一个位置的时间窗，小于第二个位置的时间窗，在第二个位置等待访问
                    wait_time = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1] - t2
                    self.wait_totalt += wait_time
                    self.cur_time = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]
                    self.total_time += add_t2 + wait_time
                    self.vehicle_loc = node_loc2
                    self.order_status[action] = 2  # 送货完成为完成状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    self.cur_load -= self.order_sets[action][3]
                    reward = -(add_t2 + wait_time)
                    distance = distance2
                    nodes = node_idx2
                elif (t2 > self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx2)[0][0]][1]):
                    # 超过第一、二个位置的时间窗，等到第二天在第第一个位置进行访问,以时间窗为准
                    wait_time = 24.0 - t1
                    self.wait_totalt += wait_time
                    self.cur_time = 0
                    self.total_time += add_t1 + wait_time
                    self.vehicle_loc = node_loc1
                    self.order_status[action] = 2  # 送货完成为完成状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    self.cur_load -= self.order_sets[action][3]
                    reward = -(add_t1 + wait_time)
                    distance = distance1
                    nodes = node_idx1
        if all(item == 2 for item in self.order_status):
            self.order_status[self.order_num-1]=1
        if self.order_status[self.order_num-1]==10:
            self.order_status[self.order_num - 1] == 2
            done = True
        else:
            done = False

        # 车辆地址换成到目的地的距离，当前车辆位置到其他可访问订单目的地的距离
        mu = []  # [1,9,1,2,2,2,3,3,3]
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                mu.append(self.order_sets[i][1])
            if element == 1:
                mu.append(self.order_sets[i][2])
            if element == 2:
                mu.append(50)
        # ord_mu = sorted(mu)#[1,2,2,2,3,3,3,9,12]
        # only_mu = list(set(ord_mu)) #下一步可访问多有节点[1,2,3,9,12 ]
        self.dismu = np.ones((2, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for i, element in enumerate(mu):
            # print(i,element)
            if element != 50:
                element_idx1 = self.node_sets[element][1]
                element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                element_loc1 = np.array([element_locx1, element_locy1])
                element_t1 = eucli(self.vehicle_loc, element_loc1) / self.vehicle_speed
                self.dismu[0][i] = element_t1
                element_idx2 = self.node_sets[element][2]
                element_locx2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][3]
                element_locy2 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx2)[0][0]][4]
                element_loc2 = np.array([element_locx2, element_locy2])
                element_t2 = eucli(self.vehicle_loc, element_loc2) / self.vehicle_speed
                self.dismu[1][i] = element_t2
            else:
                self.dismu[0][i] = 9
                self.dismu[1][i] = 9

        # 车辆负载，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu = np.repeat(0, self.order_sets.shape[0])  # [1,9,12,2,2,2,3,3,3]
        for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
            if element == 0:
                if self.order_sets[i][3] <= self.max_load - self.cur_load:
                    self.locmu[i] = -self.order_sets[i][3]
            if element == 1:
                self.locmu[i] = self.order_sets[i][3]
            if element == 2:
                self.locmu[i] = 8

        # 更新状态信息
        next_state = self.dismu,self.locmu.reshape(1,self.order_num),self.order_sta.reshape(1,self.order_num)
        return next_state, reward, done, self.total_time, self.wait_totalt, nodes, distance

    """
        每一个step后更新掩码
    """
    def upmask(self,state):
        self.mask = np.random.randint(1,2, size=(state.shape[0],state.shape[2]))
        for i in range(state.shape[0]):
            for j in range(state.shape[2]-1):
                if state[i][0][j]==0:
                    if self.order_sets[j][3]> self.max_load-self.cur_load:
                        self.mask[i][j]=0
                if state[i][0][j]==2:
                    self.mask[i][j]=0
        for i in range(state.shape[0]):
            flag = 1 #订单完成
            for j in range(state.shape[2]-1):
                if self.mask[i][j] == 1:
                    flag=0 #还有订单没完成
            if flag==0:#还有订单没完成
                self.mask[i][state.shape[2]-1] = 0#网点不可访问
            else:#所有订单完成
                self.mask[i][state.shape[2]-1] = 1 #网点可以访问
        return self.mask








#所有state是在环境里的，环境要写上初始状态信息
# next_state(时间(时间取余问题)、车辆位置、订单状态、车辆负载), reward, done = env.step(action)#约束时间选择节点哪个位置
# next_action = agent.take_action(next_state)#约束车辆负载，选择哪一个订单
# 点3：t1
# 点1：x1(时间限制),x2(时间限制)
if __name__=="__main__":
    env=Vrp_Env()
    print("订单数量:",env.order_num)
    print("订单下标id:",env.order_index)
    print("订单下标id:", env.ticoor_index)
    print("当前节点是否可以取/送货:", env.mask)
    print("当前时间:", env.cur_time)
    print("当前车辆位置:", env.vehicle_loc)
    print("当前订单状态:", env.order_status)
    print("当前车辆负载:", env.cur_load)
    # print("拼接:", env.state)
    # print("拼接:", len(env.state[2]))

    # a.append(env.state)
    # print("a",a)
    # s = np.array(a)
    # print("s", s)
    # s.reshape(len(s), -1)
    # print("s", s)
    # ss = s[:, 0].tolist()
    # print("ss", ss)
    # sss = torch.tensor(ss, dtype=torch.float)
    # print(sss.shape)
    # next_state, reward, done = env.step(5)
    # print(next_state)
    # print("当前时间:", env.cur_time)
    # print("总时间:", env.total_time)
    # print("当前车辆位置:", env.vehicle_loc)
    # print("当前订单状态:", env.order_status)
    # print("当前车辆负载:", env.cur_load)
    # print(next_state)
    # print(reward)
    # print(done)
    # print("mask前:",env.mask)
    # env.upmask((np.array([2, 3]), np.array([17]), np.array([0, 0, 0, 0, 0, 0, 0, 2, 2])))
    # print("mask后:", env.mask)