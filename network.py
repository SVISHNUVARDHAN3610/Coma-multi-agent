
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Actor1(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor1,self).__init__()
        self.state_size = state_size
        self.action = action_size
        self.fc1 = nn.Linear(self.state_size,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,512)
        self.fc5 = nn.Linear(512,128)
        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,self.action)
    def forward(self,x,i):
        s = x.shape[0]
        am = nn.Linear(s,32)
        x = am(x)
        x = self.fc7(f.relu(self.fc6(f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(f.relu(x))))))))))))
        x = f.softmax(x)
        return x

class Actor2(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor2,self).__init__()
        self.state_size = state_size
        self.action = action_size
        self.fc1 = nn.Linear(self.state_size,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,512)
        self.fc5 = nn.Linear(512,128)
        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,self.action)
    def forward(self,state,i):
        s = state.shape[0]
        am = nn.Linear(s,32)
        x = am(state)
        x = f.relu(x)
        x = self.fc7(f.relu(self.fc6(f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(x)))))))))))
        x = f.softmax(x)
        return x
        
class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Critic,self).__init__()
        self.fcs = nn.Linear(state_size,32)
        self.fca = nn.Linear(action_size,32)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,512)
        self.fc5 = nn.Linear(512,128)
        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,1)
    def forward(self,o,a,i):
        s = o.shape[0]
        am = nn.Linear(s,32)
        a1 = am(o)
        a2 = self.fca(a)
        x = torch.cat([a1.view(32,-1),a2.view(32,-1)],-1)
        x = torch.reshape(x,(-1,))
        x = self.fc7(f.relu(self.fc6(f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(f.relu(x))))))))))))
        x = f.relu(x)
        return x