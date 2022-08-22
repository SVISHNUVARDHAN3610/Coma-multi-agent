from importlib.metadata import requires
import os
import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from network import Actor1
from torch.autograd import Variable as V

class Agent1:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size= action_size
        self.gamma      = 0.999
        
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor      = Actor1(self.state_size,self.action_size)
        self.path       = ["storage/agent1/state_dist.ckpt","storage/agent1/total_model.ckpt"]
        self.optimizer  = optim.Adam(self.actor.parameters() ,lr = 0.00007)
    def choose_action(self,state,i):
        act = self.actor(state,i).to(self.device)
        return act.detach().numpy()
    def learn(self,state,next_state,reward,done,value,next_value):
        state = torch.from_numpy(state).float().to(self.device)
        next_state= torch.from_numpy(next_state).float().to(self.device)
        reward = torch.tensor(reward,dtype = torch.float32).to(self.device)
        action = self.choose_action(state,0)
        next_action = self.choose_action(next_state,1)
        action = torch.from_numpy(action).float().to(self.device)
        baseline = torch.sum(action*value)
        value = torch.tensor(value,dtype = torch.float32).to(self.device)
        advantage = torch.mean(value - baseline)
        advantage = V(advantage,requires_grad = True)
        log = V(torch.log(action),requires_grad = True)
        loss = torch.mean(advantage * log)
        torch.save(self.actor.state_dict(),self.path[0])
        torch.save(self.actor,self.path[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss