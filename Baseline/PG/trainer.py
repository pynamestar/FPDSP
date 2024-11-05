import copy

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
writer = SummaryWriter('/root/tf-logs/vrppg30dim256')
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    losses=[]
    totime_list = []
    waittime_list = []

    nodes_list = []
    distance_list=[]
    maxre = -10000
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i,ncols=100) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                distance_return=0
                nodes = []
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                total_time1=[]
                wait_totalt1=[]
                nodes1=[]
                distance_return1=[]
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done,total_time,wait_totalt,vehicle_loc,distance = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = copy.deepcopy(next_state)
                    episode_return += reward
                    nodes.append(vehicle_loc)
                    distance_return += distance
                    # env.upmask(state)

                # 保存最大epoisde奖励的参数
                if maxre < episode_return:
                    maxre = episode_return
                    agent.save()
                # 添加标量
                writer.add_scalar(tag="reward", scalar_value=episode_return,
                                  global_step=i * num_episodes/10+ i_episode)
                return_list.append(episode_return)
                loss=agent.update(transition_dict)
                losses.append(loss.detach().cpu().numpy())
                totime_list.append(total_time)
                waittime_list.append(wait_totalt)
                nodes_list.append(nodes)
                distance_list.append(distance_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

                total_time1.append(total_time)
                wait_totalt1.append(wait_totalt)
                nodes1.append(nodes)
                distance_return1.append(distance_return)
                # totime_list1 = np.array(total_time1)
                # data2 = pd.DataFrame(totime_list1)
                # data2.to_csv('totime_list1.csv',mode='a',index=False,header=False)
                #
                # waittime_list1 = np.array(wait_totalt1)
                # data3 = pd.DataFrame(waittime_list1)
                # data3.to_csv('waittime_list1.csv',mode='a',index=False,header=False)
                #
                # nodes_list1 = np.array(nodes1)
                # data4 = pd.DataFrame(nodes_list1)
                # data4.to_csv('nodes_list1.csv',mode='a',index=False,header=False)
                #
                # distance_list1 = np.array(distance_return1)
                # data5 = pd.DataFrame(distance_list1)
                # data5.to_csv('distance_list1.csv',mode='a',index=False,header=False)

    return return_list,losses,totime_list,waittime_list,nodes_list,distance_list

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

#优势估计
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)