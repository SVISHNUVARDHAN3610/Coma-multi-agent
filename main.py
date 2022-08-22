import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from make_env import make_env
from torch.autograd import Variable as V
from network import Critic
from Agents.agent1 import Agent1
from Agents.agent2 import Agent2
from storage.buffer import Buffer
from storage.plt import Plt
plt = Plt()
env = make_env("simple_reference")
buffer = Buffer()


class Main:
    def __init__(self,state_size,action_size,n_games,step):
        self.state_size = state_size
        self.action_size = action_size
        self.n_games = n_games
        self.step = step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.agent1 = Agent1(self.state_size,self.action_size)
        self.agent2 = Agent2(self.state_size,self.action_size)
        self.critic = Critic(self.state_size,self.action_size).to(self.device)
        self.path   = ["storage/centeral_critic/state_dist.ckpt","storage/centeral_critic/total_model.ckpt"]
        self.optim  = optim.Adam(self.critic.parameters(),lr = 0.000009)
    def choose_action(self,state,i):
        actions = []
        obs1 = torch.from_numpy(state[0]).float().to(self.device)
        obs2 = torch.from_numpy(state[1]).float().to(self.device)
        act1 = self.agent1.choose_action(obs1,i)
        act2 = self.agent2.choose_action(obs2,i)
        actions.append(act1)
        actions.append(act2)
        return actions
    def appending(self,qvalue,nvalue,loss,state,next_state,l1,l2,p1,p2):
        buffer.agent1_qvalue.append(qvalue[0])
        buffer.agent2_qvalue.append(qvalue[1])
        buffer.agent1_nvalue.append(nvalue[0])
        buffer.agent2_nvalue.append(nvalue[1])
        buffer.loss.append(loss.detach().numpy())
        buffer.agent1_state.append(state[0])
        buffer.agent2_state.append(state[1])
        buffer.agent1_next_state.append(next_state[0])
        buffer.agent2_next_state.append(next_state[1])
        buffer.agent1_loss.append(l1.detach().numpy())
        buffer.agent2_loss.append(l2.detach().numpy())
        buffer.agent1_policy.append(p1)
        buffer.agent2_policy.append(p2)
    def q_value(self,state,next_state,action,next_action):
        q_value = []
        next_qvalue = []
        obs1 = torch.from_numpy(state[0]).float().to(self.device)
        obs2 = torch.from_numpy(state[1]).float().to(self.device)
        nobs1 = torch.from_numpy(next_state[0]).float().to(self.device)
        nobs2 = torch.from_numpy(next_state[1]).float().to(self.device)
        q1 = self.critic(obs1,torch.tensor(action[0],dtype = torch.float32).to(self.device),0).to(self.device)
        q2 = self.critic(obs2,torch.tensor(action[1],dtype = torch.float32).to(self.device),0).to(self.device)
        nq1 = self.critic(nobs1,torch.tensor(next_action[0],dtype = torch.float32).to(self.device),1).to(self.device)
        nq2 = self.critic(nobs2,torch.tensor(next_action[1],dtype = torch.float32).to(self.device),1).to(self.device)
        q_value.append(q1.detach().numpy())
        q_value.append(q2.detach().numpy())
        next_qvalue.append(nq1.detach().numpy())
        next_qvalue.append(nq2.detach().numpy())
        return q_value,next_qvalue
    def saving(self):
        torch.save(self.critic.state_dict() , self.path[0])
        torch.save(self.critic,self.path[1])
        #print("saving..............................................")
    def learn(self,state,next_state,reward,done,i):
        action = self.choose_action(state,0)
        next_action = self.choose_action(next_state,1)
        value,next_value = self.q_value(state,next_state,action,next_action)
        returns = sum(reward) + self.gamma*sum(next_value) - sum(value)
        returns = torch.tensor(returns,dtype =torch.float32).to(self.device)
        returns = V(returns,requires_grad = True)
        vv  = torch.tensor(sum(value),dtype = torch.float32).to(self.device)
        mv      = V(vv,requires_grad = True)
        loss = torch.mean((returns - mv)**2)
        self.saving()
        k1 = self.agent1.learn(state[0],next_state[0],reward[0],done[0],value[0],next_value[0])
        k2 = self.agent2.learn(state[1],next_state[1],reward[1],done[1],value[1],next_value[1])
        self.appending(value,next_value,loss,state,next_state,k1,k2,torch.log(torch.tensor(action[0])),torch.log(torch.tensor(action[0])))
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
    def clear(self):
        buffer.agent1_reward = []
        buffer.agent2_reward = []
        buffer.agent1_loss = []
        buffer.agent2_loss = []
        buffer.agent1_policy= []
        buffer.agent2_policy= []
        buffer.agent1_qvalue = []
        buffer.agent2_qvalue = []
        buffer.agent1_nvalue = []
        buffer.agent2_qvalue = []
        buffer.loss = []
    def save(self):
        plt.ag1reward(buffer.agent1_mean,buffer.episodes)
        plt.ag2reward(buffer.agent2_mean,buffer.episodes)
        plt.ag1qvalue(buffer.agent1_qmean,buffer.episodes)
        plt.ag2qvalue(buffer.agent2_qmean,buffer.episodes)
        plt.ag1nqvalue(buffer.agent1_nqmean,buffer.episodes)
        plt.ag2nqvalue(buffer.agent2_nqmean,buffer.episodes)
        plt.pltag1loss(buffer.agent1_meanloss,buffer.episodes)
        plt.pltag2loss(buffer.agent2_meanloss,buffer.episodes)
        plt.pltloss(buffer.meanloss,buffer.episodes)
    def ever_episode(self,i):
        if i==0:
            a1mean = sum(buffer.agent1_reward)/1
            a2mean = sum(buffer.agent2_reward)/1
            a1loss = sum(buffer.agent1_loss)
            a2loss = sum(buffer.agent2_loss)
            a1policy = sum(buffer.agent1_policy)
            a2policy = sum(buffer.agent2_policy)
            a1qmean = sum(buffer.agent1_qvalue)
            a2qmean = sum(buffer.agent2_qvalue)
            a1nmean = sum(buffer.agent1_nvalue)
            a2nmean = sum(buffer.agent2_nvalue)
            loss = sum(buffer.loss)
        else:
            a1mean = sum(buffer.agent1_reward)/len(buffer.agent1_reward)
            a2mean = sum(buffer.agent2_reward)/len(buffer.agent2_reward)
            a1loss = sum(buffer.agent1_loss)/len(buffer.agent1_loss)
            a2loss = sum(buffer.agent2_loss)/len(buffer.agent2_loss)
            a1policy = sum(buffer.agent1_policy)/len(buffer.agent1_policy)
            a2policy = sum(buffer.agent2_policy)/len(buffer.agent1_policy)
            a1qmean = sum(buffer.agent1_qvalue)/len(buffer.agent1_qvalue)
            a2qmean = sum(buffer.agent2_qvalue)/len(buffer.agent2_qvalue)
            a1nmean = sum(buffer.agent1_nvalue)/len(buffer.agent1_nvalue)
            a2nmean = sum(buffer.agent2_nvalue)/len(buffer.agent1_nvalue)
            loss = sum(buffer.loss)/len(buffer.loss)
        buffer.agent1_mean.append(a1mean)
        buffer.agent2_mean.append(a2mean)
        buffer.agent1_meanloss.append(a1loss)
        buffer.agent2_meanloss.append(a2loss)
        buffer.agent1_mean_policy.append(a1policy)
        buffer.agent2_mean_policy.append(a2policy)
        buffer.agent1_qmean.append(a1qmean)
        buffer.agent2_qmean.append(a2qmean)
        buffer.agent1_nqmean.append(a1nmean)
        buffer.agent2_nqmean.append(a2nmean)
        buffer.meanloss.append(loss)

    def play(self):
        for i in range(self.n_games):
            state = env.reset()
            score = [0*env.n]
            done  = [False * 2]
            buffer.episodes.append(i)
            self.ever_episode(i)
            self.save()
            self.clear()
            print("episode:",i,"agent1_reward:",buffer.agent1_mean[i],"agent2_reward:",buffer.agent2_mean[i])
            for _ in range(self.step):
                action = self.choose_action(state,0)
                next_state,reward,done,_ = env.step(action)
                if done:
                    self.learn(state,next_state,reward,done,i)
                    score += reward
                    state = next_state
                    buffer.agent1_reward.append(reward[0])
                    buffer.agent2_reward.append(reward[1]) 
                else: 
                    self.learn(state,next_state,reward,done,i)
                    score += reward
                    state = next_state
                    buffer.agent1_reward.append(reward[0])
                    buffer.agent2_reward.append(reward[1])      
    
if __name__ =="__main__":
    main = Main(21,5,1500,300)
    main.play()